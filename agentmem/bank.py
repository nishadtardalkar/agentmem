from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from agentmem.chunker import Episode


@dataclass
class MemoryEntry:
    """One stored episode (the memory value)."""

    role: str
    text: str
    sentences: list[str]
    ts: str


@dataclass
class RetrievedEpisode:
    """An episode retrieved via the token-key index."""

    episode_id: str
    entry: MemoryEntry
    score: float = 1.0

    # Back-compat aliases used by readout / HTTP layer.
    @property
    def latent_id(self) -> str:
        return self.episode_id

    @property
    def entries(self) -> list[MemoryEntry]:
        return [self.entry]


# Back-compat name
RetrievedBucket = RetrievedEpisode


class EpisodeBank:
    """Store: episode values. Index: FAISS token keys → episode_id."""

    def __init__(self, dim: int, data_dir: Path) -> None:
        self.dim = dim
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_path = self.data_dir / "keys.faiss"
        self.sqlite_path = self.data_dir / "episodes.db"
        self.id_map_path = self.data_dir / "faiss_ids.json"

        self._index = self._load_or_create_index()
        self._faiss_id_to_episode: dict[int, str] = self._load_id_map()
        self._next_faiss_id = (
            (max(self._faiss_id_to_episode.keys()) + 1)
            if self._faiss_id_to_episode
            else 0
        )
        self._init_db()

    def _load_or_create_index(self) -> faiss.Index:
        if self.faiss_path.exists():
            index = faiss.read_index(str(self.faiss_path))
            if index.d != self.dim:
                raise ValueError(
                    f"FAISS dim {index.d} does not match encoder dim {self.dim}"
                )
            return index
        base = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIDMap2(base)

    def _load_id_map(self) -> dict[int, str]:
        if not self.id_map_path.exists():
            return {}
        raw = json.loads(self.id_map_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}

    def _save_id_map(self) -> None:
        payload = {str(k): v for k, v in self._faiss_id_to_episode.items()}
        self.id_map_path.write_text(json.dumps(payload), encoding="utf-8")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            # Drop legacy shapes (merged latent buckets / flat-incompatible).
            conn.execute("DROP TABLE IF EXISTS latents")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sentences_json TEXT NOT NULL,
                    ts TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self) -> None:
        faiss.write_index(self._index, str(self.faiss_path))
        self._save_id_map()

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def _add_keys(self, episode_id: str, keys: np.ndarray) -> None:
        if keys.size == 0:
            return
        keys = np.ascontiguousarray(keys.astype(np.float32))
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)
        n = keys.shape[0]
        ids = np.arange(self._next_faiss_id, self._next_faiss_id + n, dtype=np.int64)
        self._index.add_with_ids(keys, ids)
        for fid in ids:
            self._faiss_id_to_episode[int(fid)] = episode_id
        self._next_faiss_id += n

    def _get_episode_row(self, episode_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            )
            return cur.fetchone()

    @staticmethod
    def _entry_from_episode(episode: Episode) -> MemoryEntry:
        return MemoryEntry(
            role=episode.role,
            text=episode.text,
            sentences=list(episode.sentences),
            ts=episode.timestamp,
        )

    @staticmethod
    def _entry_from_row(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            role=row["role"],
            text=row["text"],
            sentences=list(json.loads(row["sentences_json"])),
            ts=row["ts"],
        )

    def _row_to_retrieved(
        self, row: sqlite3.Row, *, score: float = 1.0
    ) -> RetrievedEpisode:
        return RetrievedEpisode(
            episode_id=row["episode_id"],
            entry=self._entry_from_row(row),
            score=score,
        )

    def store(self, episode: Episode) -> str:
        """Always insert the episode as a new value; index its token keys to it.

        Returns the episode_id.
        """
        episode_id = episode.episode_id or str(uuid.uuid4())
        entry = self._entry_from_episode(episode)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes (episode_id, role, text, sentences_json, ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    entry.role,
                    entry.text,
                    json.dumps(entry.sentences),
                    entry.ts,
                ),
            )
            conn.commit()
        self._add_keys(episode_id, episode.keys)
        self.save()
        return episode_id

    # Back-compat name used by session / older call sites.
    def upsert(self, episode: Episode) -> str:
        return self.store(episode)

    def search(
        self,
        query_keys: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.70,
    ) -> list[RetrievedEpisode]:
        if self._index.ntotal == 0 or query_keys.size == 0:
            return []

        query_keys = np.ascontiguousarray(query_keys.astype(np.float32))
        if query_keys.ndim == 1:
            query_keys = query_keys.reshape(1, -1)

        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(query_keys, k)

        best_by_episode: dict[str, float] = {}
        for row_scores, row_ids in zip(scores, ids):
            for score, fid in zip(row_scores, row_ids):
                fid_i = int(fid)
                if fid_i < 0:
                    continue
                sc = float(score)
                if sc < threshold:
                    continue
                episode_id = self._faiss_id_to_episode.get(fid_i)
                if episode_id is None:
                    continue
                prev = best_by_episode.get(episode_id)
                if prev is None or sc > prev:
                    best_by_episode[episode_id] = sc

        results: list[RetrievedEpisode] = []
        for episode_id, score in sorted(
            best_by_episode.items(), key=lambda x: x[1], reverse=True
        ):
            row = self._get_episode_row(episode_id)
            if row is None:
                continue
            results.append(self._row_to_retrieved(row, score=score))
            if len(results) >= top_k:
                break
        return results

    def get_episode(self, episode_id: str) -> RetrievedEpisode | None:
        row = self._get_episode_row(episode_id)
        if row is None:
            return None
        return self._row_to_retrieved(row)

    def get_latent(self, latent_id: str) -> RetrievedEpisode | None:
        return self.get_episode(latent_id)

    def list_episodes(self) -> list[RetrievedEpisode]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY ts ASC"
            ).fetchall()
        return [self._row_to_retrieved(row) for row in rows]

    def list_latents(self) -> list[RetrievedEpisode]:
        return self.list_episodes()

    def episode_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()
        return int(row["n"])

    def latent_count(self) -> int:
        return self.episode_count()
