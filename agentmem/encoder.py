from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    """Produces L2-normalized latent vectors from text."""

    @property
    def dim(self) -> int: ...

    def encode(self, text: str) -> np.ndarray: ...

    def encode_many(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if vectors.ndim == 1:
        norm = float(np.linalg.norm(vectors))
        return vectors / max(norm, eps)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


class HiddenStateEncoder:
    """Masked sum of hidden states from a small causal LM (bf16 by default).

    For short keys like ``dog's``, summing subword states keeps the noun signal
    instead of last-token pooling onto ``'s``. After L2-normalize, sum == mean.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        layer: int = -1,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.layer = layer
        self.device = device
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        # Resolve hub id -> local snapshot so load never contacts the network.
        model_path = snapshot_download(model_id, local_files_only=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to(device)
        self.model.eval()
        self._dim = int(self.model.config.hidden_size)
        self._torch = torch

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[self.layer]  # (B, T, D)
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)  # (B, D)
            vectors = summed.float().cpu().numpy()

        return l2_normalize(vectors).astype(np.float32)


def _stable_seed(text: str) -> int:
    import hashlib

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class HashEncoder:
    """Deterministic fake encoder for tests (no torch model required)."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]

    def encode_many(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(_stable_seed(text))
            vec = rng.standard_normal(self._dim).astype(np.float32)
            # Mix in a bag-of-words signal so similar sentences can be close
            for token in text.lower().split():
                token_rng = np.random.default_rng(_stable_seed(token))
                vec += 0.35 * token_rng.standard_normal(self._dim).astype(np.float32)
            out[i] = vec
        return l2_normalize(out).astype(np.float32)
