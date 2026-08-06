from __future__ import annotations

import json
import re
from typing import Protocol, Sequence

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "by",
        "with",
        "from",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
        "not",
        "no",
        "yes",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "out",
        "up",
        "down",
        "also",
        "there",
        "here",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
_JSON_LIST_RE = re.compile(r"\[[\s\S]*?\]")

_EXTRACT_SYSTEM = (
    "Extract the most important key tokens for memory retrieval from the text. "
    "Return ONLY a JSON array of short lowercase strings (entities, topics, "
    "salient nouns). No prose."
)


class KeyExtractor(Protocol):
    """Produces short key strings used as FAISS index terms."""

    def extract(self, text: str) -> list[str]: ...

    def extract_many(self, texts: Sequence[str]) -> list[list[str]]: ...


def heuristic_keys(text: str, max_keys: int = 8) -> list[str]:
    """Deterministic fallback: lowercase non-stopword tokens, order-preserving unique."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _WORD_RE.findall(text.lower()):
        if match in _STOPWORDS or len(match) < 2:
            continue
        if match in seen:
            continue
        seen.add(match)
        out.append(match)
        if len(out) >= max_keys:
            break
    return out


def _parse_key_list(raw: str, max_keys: int) -> list[str] | None:
    raw = raw.strip()
    if not raw:
        return None
    candidates = [raw]
    match = _JSON_LIST_RE.search(raw)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        keys: list[str] = []
        seen: set[str] = set()
        for item in data:
            if not isinstance(item, str):
                continue
            token = item.strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            keys.append(token)
            if len(keys) >= max_keys:
                break
        if keys:
            return keys
    return None


class HeuristicKeyExtractor:
    """GPU-free extractor for tests / fallback."""

    def __init__(self, max_keys: int = 8) -> None:
        self.max_keys = max_keys

    def extract(self, text: str) -> list[str]:
        return heuristic_keys(text, max_keys=self.max_keys)

    def extract_many(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.extract(t) for t in texts]


class ModelKeyExtractor:
    """Ask the encoder instruct model for a JSON list of key tokens."""

    def __init__(self, encoder, max_keys: int = 8, max_new_tokens: int = 64) -> None:
        self.encoder = encoder
        self.max_keys = max_keys
        self.max_new_tokens = max_new_tokens
        self._fallback = HeuristicKeyExtractor(max_keys=max_keys)

    def extract(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        try:
            raw = self._generate(text)
            parsed = _parse_key_list(raw, self.max_keys)
            if parsed:
                return parsed
        except Exception:
            pass
        return self._fallback.extract(text)

    def extract_many(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.extract(t) for t in texts]

    def _generate(self, text: str) -> str:
        torch = self.encoder._torch
        tokenizer = self.encoder.tokenizer
        model = self.encoder.model
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.encoder.device)
        attention_mask = encoded["attention_mask"].to(self.encoder.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0, input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
