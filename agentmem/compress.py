from __future__ import annotations

from typing import Protocol


_COMPRESS_SYSTEM = (
    "Compress this text into a dense memory note for later retrieval. "
    "Keep all entities, roles, relationships, and facts. "
    "Use short declarative clauses; drop filler, hedging, and repetition. "
    "Do not invent details. Prefer names and roles over pronouns. "
    "Target: at most 1 sentence per distinct fact; total under 40 words. "
    "Return ONLY the compressed note as plain text. No bullets, labels, or explanation."
)


class EpisodeCompressor(Protocol):
    """Rewrites episode text into a dense note before storage."""

    def compress(self, text: str) -> str: ...


class ModelEpisodeCompressor:
    """Ask the encoder instruct model for a dense memory note."""

    def __init__(self, encoder) -> None:
        self.encoder = encoder

    def compress(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        try:
            raw = self._generate(text)
        except Exception:
            return text
        out = raw.strip()
        return out or text

    def _generate(self, text: str) -> str:
        torch = self.encoder._torch
        tokenizer = self.encoder.tokenizer
        model = self.encoder.model
        messages = [
            {"role": "system", "content": _COMPRESS_SYSTEM},
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
