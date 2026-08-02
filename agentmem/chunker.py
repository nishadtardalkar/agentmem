from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from agentmem.encoder import Encoder

Role = Literal["user", "assistant"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Episode:
    """A semantically coherent span of a half-turn."""

    role: Role
    text: str
    sentences: list[str]
    latents: np.ndarray  # (N, D) one row per sentence, L2-normalized
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    episode_id: str | None = None

    @property
    def keys(self) -> np.ndarray:
        return self.latents


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


@dataclass
class SemanticChunker:
    """Sentence-split a half-turn and break/merge by adjacent latent similarity."""

    encoder: Encoder
    tau_break: float = 0.75

    def chunk(self, text: str, role: Role) -> list[Episode]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        latents = self.encoder.encode_many(sentences)
        if len(sentences) == 1:
            return [
                Episode(
                    role=role,
                    text=sentences[0],
                    sentences=sentences,
                    latents=latents,
                )
            ]

        spans: list[tuple[int, int]] = []
        start = 0
        for i in range(len(sentences) - 1):
            if cosine(latents[i], latents[i + 1]) < self.tau_break:
                spans.append((start, i + 1))
                start = i + 1
        spans.append((start, len(sentences)))

        episodes: list[Episode] = []
        for lo, hi in spans:
            span_sents = sentences[lo:hi]
            span_z = latents[lo:hi]
            episodes.append(
                Episode(
                    role=role,
                    text=" ".join(span_sents),
                    sentences=list(span_sents),
                    latents=span_z.copy(),
                )
            )
        return episodes

    def query_keys(self, text: str) -> np.ndarray:
        """Encode each sentence of a query half-turn for retrieval."""
        sentences = split_sentences(text)
        if not sentences:
            return np.zeros((0, self.encoder.dim), dtype=np.float32)
        return self.encoder.encode_many(sentences)
