from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import numpy as np

from agentmem.encoder import Encoder
from agentmem.keys import HeuristicKeyExtractor, KeyExtractor

Role = Literal["user", "assistant"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Episode:
    """A semantically coherent span of a half-turn."""

    role: Role
    text: str
    sentences: list[str]
    keys: np.ndarray  # (K, D) extracted token embeddings, L2-normalized
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    episode_id: str | None = None


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
    """Sentence-split a half-turn; break on sentence cosine; index via key tokens."""

    encoder: Encoder
    key_extractor: KeyExtractor = field(default_factory=HeuristicKeyExtractor)
    tau_break: float = 0.75

    def _token_keys_for_texts(self, texts: list[str]) -> np.ndarray:
        """Extract key tokens per text, embed, and stack unique rows."""
        all_tokens: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for token in self.key_extractor.extract(text):
                if token in seen:
                    continue
                seen.add(token)
                all_tokens.append(token)
        if not all_tokens:
            return np.zeros((0, self.encoder.dim), dtype=np.float32)
        return self.encoder.encode_many(all_tokens)

    def chunk(self, text: str, role: Role) -> list[Episode]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        # Sentence embeddings drive tau_break only; FAISS keys are token embeddings.
        sentence_latents = self.encoder.encode_many(sentences)
        if len(sentences) == 1:
            return [
                Episode(
                    role=role,
                    text=sentences[0],
                    sentences=sentences,
                    keys=self._token_keys_for_texts(sentences),
                )
            ]

        spans: list[tuple[int, int]] = []
        start = 0
        for i in range(len(sentences) - 1):
            if cosine(sentence_latents[i], sentence_latents[i + 1]) < self.tau_break:
                spans.append((start, i + 1))
                start = i + 1
        spans.append((start, len(sentences)))

        episodes: list[Episode] = []
        for lo, hi in spans:
            span_sents = sentences[lo:hi]
            episodes.append(
                Episode(
                    role=role,
                    text=" ".join(span_sents),
                    sentences=list(span_sents),
                    keys=self._token_keys_for_texts(span_sents),
                )
            )
        return episodes

    def query_keys(self, text: str) -> np.ndarray:
        """Extract key tokens from a query half-turn and embed them."""
        sentences = split_sentences(text)
        if not sentences:
            return self._token_keys_for_texts([text] if text.strip() else [])
        return self._token_keys_for_texts(sentences)
