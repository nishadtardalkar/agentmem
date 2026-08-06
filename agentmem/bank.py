from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from agentmem.chunker import Episode


@dataclass
class MemoryEntry:
    """One memory object stored under a latent bucket."""

    role: str
    text: str
    sentences: list[str]
    ts: str


@dataclass
class RetrievedBucket:
    """A latent bucket and the memory objects that fall into it."""

    latent_id: str
    entries: list[MemoryEntry]
    score: float = 1.0


class EpisodeBank:
    """FAISS sentence keys → latent_id → [MemoryEntry, ...]."""

    def __init__(self, dim: int, data_dir: Path, tau_upsert: float = 0.75) -> None:
        self.dim = dim
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tau_upsert = tau_upsert

        self.faiss_path = self.data_dir / "keys.faiss"
        self.sqlite_path = self.data_dir / "episodes.db"
        self.id_map_path = self.data_dir / "faiss_ids.json"

        self._index = self._load_or_create_index()
        self._faiss_id_to_latent: dict[int, str] = self._load_id_map()
        self._next_faiss_id = (
            (max(self._faiss_id_to_latent.keys()) + 1) if self._faiss_id_to_latent else 0
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
        payload = {str(k): v for k, v in self._faiss_id_to_latent.items()}
        self.id_map_path.write_text(json.dumps(payload), encoding="utf-8")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS latents (
                    latent_id TEXT PRIMARY KEY,
                    entries_json TEXT NOT NULL
                )
                """
            )
            # Drop legacy flat-episode table if present (incompatible shape).
            conn.execute("DROP TABLE IF EXISTS episodes")
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
        best_latent: str | None = None
        best_score = -1.0
        for row_scores, row_ids in zip(scores, ids):
            fid = int(row_ids[0])
            score = float(row_scores[0])
            if fid < 0:
                continue
            if score > best_score:
                best_score = score
                best_latent = self._faiss_id_to_latent.get(fid)
        return best_latent, best_score

    def _add_keys(self, latent_id: str, keys: np.ndarray) -> None:
        if keys.size == 0:
            return
        keys = np.ascontiguousarray(keys.astype(np.float32))
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)
        n = keys.shape[0]
        ids = np.arange(self._next_faiss_id, self._next_faiss_id + n, dtype=np.int64)
        self._index.add_with_ids(keys, ids)
        for fid in ids:
            self._faiss_id_to_latent[int(fid)] = latent_id
        self._next_faiss_id += n

    def _get_latent_row(self, latent_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT * FROM latents WHERE latent_id = ?", (latent_id,)
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
    def _parse_entries(raw: str) -> list[MemoryEntry]:
        data = json.loads(raw)
        return [
            MemoryEntry(
                role=item["role"],
                text=item["text"],
                sentences=list(item["sentences"]),
                ts=item["ts"],
            )
            for item in data
        ]

    @staticmethod
    def _dump_entries(entries: list[MemoryEntry]) -> str:
        return json.dumps([asdict(e) for e in entries])

    def _row_to_bucket(
        self, row: sqlite3.Row, *, score: float = 1.0
    ) -> RetrievedBucket:
        return RetrievedBucket(
            latent_id=row["latent_id"],
            entries=self._parse_entries(row["entries_json"]),
            score=score,
        )

    def upsert(self, episode: Episode) -> str:
        """Insert a new latent bucket or append an entry into a similar one.

        Returns the latent_id of the bucket that received the entry.
        """
        keys = episode.keys
        match_id, best_sim = self._best_match(keys)
        entry = self._entry_from_episode(episode)

        if match_id is not None and best_sim >= self.tau_upsert:
            row = self._get_latent_row(match_id)
            if row is None:
                # stale map entry; fall through to insert
                pass
            else:
                entries = self._parse_entries(row["entries_json"])
                entries.append(entry)
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE latents
                        SET entries_json = ?
                        WHERE latent_id = ?
                        """,
                        (self._dump_entries(entries), match_id),
                    )
                    conn.commit()
                self._add_keys(match_id, keys)
                self.save()
                return match_id

        latent_id = episode.episode_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO latents (latent_id, entries_json)
                VALUES (?, ?)
                """,
                (latent_id, self._dump_entries([entry])),
            )
            conn.commit()
        self._add_keys(latent_id, keys)
        self.save()
        return latent_id

    def search(
        self,
        query_keys: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.70,
    ) -> list[RetrievedBucket]:
        if self._index.ntotal == 0 or query_keys.size == 0:
            return []

        query_keys = np.ascontiguousarray(query_keys.astype(np.float32))
        if query_keys.ndim == 1:
            query_keys = query_keys.reshape(1, -1)

        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(query_keys, k)

        best_by_latent: dict[str, float] = {}
        for row_scores, row_ids in zip(scores, ids):
            for score, fid in zip(row_scores, row_ids):
                fid_i = int(fid)
                if fid_i < 0:
                    continue
                sc = float(score)
                if sc < threshold:
                    continue
                latent_id = self._faiss_id_to_latent.get(fid_i)
                if latent_id is None:
                    continue
                prev = best_by_latent.get(latent_id)
                if prev is None or sc > prev:
                    best_by_latent[latent_id] = sc

        results: list[RetrievedBucket] = []
        for latent_id, score in sorted(
            best_by_latent.items(), key=lambda x: x[1], reverse=True
        ):
            row = self._get_latent_row(latent_id)
            if row is None:
                continue
            results.append(self._row_to_bucket(row, score=score))
            if len(results) >= top_k:
                break
        return results

    def get_latent(self, latent_id: str) -> RetrievedBucket | None:
        row = self._get_latent_row(latent_id)
        if row is None:
            return None
        return self._row_to_bucket(row)

    # Back-compat aliases used by the HTTP API / debug REPL.
    def get_episode(self, latent_id: str) -> RetrievedBucket | None:
        return self.get_latent(latent_id)

    def list_latents(self) -> list[RetrievedBucket]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM latents").fetchall()
        buckets = [self._row_to_bucket(row) for row in rows]
        # Order by earliest entry timestamp within each bucket.
        buckets.sort(
            key=lambda b: b.entries[0].ts if b.entries else "",
        )
        return buckets

    def list_episodes(self) -> list[RetrievedBucket]:
        return self.list_latents()

    def latent_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM latents").fetchone()
        return int(row["n"])

    def episode_count(self) -> int:
        return self.latent_count()
