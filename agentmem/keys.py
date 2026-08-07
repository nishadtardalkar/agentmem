from __future__ import annotations

import json
import re
from typing import Protocol, Sequence

_JSON_LIST_RE = re.compile(r"\[[\s\S]*?\]")

_EXTRACT_SYSTEM = (
    "Extract key tokens for memory retrieval from the text. "
    "Include the subject (including pronouns like i/you/he/she/we/they), "
    "main verbs, and salient nouns/entities/places. "
    "Drop function words (articles, prepositions, auxiliaries, conjunctions). "
    "Return ONLY a JSON array of short lowercase strings. No prose."
)


class KeyExtractor(Protocol):
    """Produces short key strings used as FAISS index terms."""

    def extract(self, text: str) -> list[str]: ...

    def extract_many(self, texts: Sequence[str]) -> list[list[str]]: ...


def _parse_key_list(raw: str) -> list[str] | None:
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
        if keys:
            return keys
    return None


class ModelKeyExtractor:
    """Ask the encoder instruct model for a JSON list of key tokens."""

    def __init__(self, encoder) -> None:
        self.encoder = encoder

    def extract(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        try:
            raw = self._generate(text)
        except Exception:
            return []
        return _parse_key_list(raw) or []

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
        # Fill remaining context; stop on EOS (no fixed reply length cap).
        ctx = int(getattr(model.config, "max_position_embeddings", 32768))
        max_new_tokens = max(1, ctx - int(input_ids.shape[-1]))
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0, input_ids.shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
