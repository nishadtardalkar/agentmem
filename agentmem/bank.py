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
class RetrievedEpisode:
    episode_id: str
    role: str
    text: str
    sentences: list[str]
    timestamp: str
    score: float


class EpisodeBank:
    """FAISS sentence keys → SQLite full-episode values."""

    def __init__(self, dim: int, data_dir: Path, tau_upsert: float = 0.75) -> None:
        self.dim = dim
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tau_upsert = tau_upsert

        self.faiss_path = self.data_dir / "keys.faiss"
        self.sqlite_path = self.data_dir / "episodes.db"
        self.id_map_path = self.data_dir / "faiss_ids.json"

        self._index = self._load_or_create_index()
        self._faiss_id_to_episode: dict[int, str] = self._load_id_map()
        self._next_faiss_id = (
            (max(self._faiss_id_to_episode.keys()) + 1) if self._faiss_id_to_episode else 0
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sentences_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
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

    def _best_match(self, keys: np.ndarray) -> tuple[str | None, float]:
        if self._index.ntotal == 0 or keys.size == 0:
            return None, -1.0

        keys = np.ascontiguousarray(keys.astype(np.float32))
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)

        scores, ids = self._index.search(keys, 1)
        best_ep: str | None = None
        best_score = -1.0
        for row_scores, row_ids in zip(scores, ids):
            fid = int(row_ids[0])
            score = float(row_scores[0])
            if fid < 0:
                continue
            if score > best_score:
                best_score = score
                best_ep = self._faiss_id_to_episode.get(fid)
        return best_ep, best_score

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
    def _stamped_value(episode: Episode) -> str:
        """Episode value text with timestamp baked in."""
        return f"[{episode.timestamp}] {episode.text}".strip()

    def upsert(self, episode: Episode) -> str:
        """Insert a new episode or append into a similar existing one. Returns episode_id."""
        keys = episode.keys
        match_id, best_sim = self._best_match(keys)
        stamped = self._stamped_value(episode)

        if match_id is not None and best_sim >= self.tau_upsert:
            row = self._get_episode_row(match_id)
            if row is None:
                # stale map entry; fall through to insert
                pass
            else:
                existing_sents = json.loads(row["sentences_json"])
                new_sents = list(episode.sentences)
                merged_sents = existing_sents + new_sents
                merged_text = (row["text"].rstrip() + " " + stamped).strip()
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE episodes
                        SET text = ?, sentences_json = ?, timestamp = ?
                        WHERE episode_id = ?
                        """,
                        (
                            merged_text,
                            json.dumps(merged_sents),
                            episode.timestamp,
                            match_id,
                        ),
                    )
                    conn.commit()
                self._add_keys(match_id, keys)
                self.save()
                return match_id

        episode_id = episode.episode_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes (episode_id, role, text, sentences_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    episode.role,
                    stamped,
                    json.dumps(list(episode.sentences)),
                    episode.timestamp,
                ),
            )
            conn.commit()
        self._add_keys(episode_id, keys)
        self.save()
        return episode_id

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
                ep_id = self._faiss_id_to_episode.get(fid_i)
                if ep_id is None:
                    continue
                prev = best_by_episode.get(ep_id)
                if prev is None or sc > prev:
                    best_by_episode[ep_id] = sc

        results: list[RetrievedEpisode] = []
        for ep_id, score in sorted(
            best_by_episode.items(), key=lambda x: x[1], reverse=True
        ):
            row = self._get_episode_row(ep_id)
            if row is None:
                continue
            results.append(
                RetrievedEpisode(
                    episode_id=ep_id,
                    role=row["role"],
                    text=row["text"],
                    sentences=json.loads(row["sentences_json"]),
                    timestamp=row["timestamp"],
                    score=score,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def get_episode(self, episode_id: str) -> RetrievedEpisode | None:
        row = self._get_episode_row(episode_id)
        if row is None:
            return None
        return RetrievedEpisode(
            episode_id=episode_id,
            role=row["role"],
            text=row["text"],
            sentences=json.loads(row["sentences_json"]),
            timestamp=row["timestamp"],
            score=1.0,
        )

    def list_episodes(self) -> list[RetrievedEpisode]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp ASC"
            ).fetchall()
        return [
            RetrievedEpisode(
                episode_id=row["episode_id"],
                role=row["role"],
                text=row["text"],
                sentences=json.loads(row["sentences_json"]),
                timestamp=row["timestamp"],
                score=1.0,
            )
            for row in rows
        ]

    def episode_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()
        return int(row["n"])
