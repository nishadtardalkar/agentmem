from __future__ import annotations

from agentmem.bank import EpisodeBank
from agentmem.chunker import SemanticChunker
from agentmem.config import MemoryConfig
from agentmem.encoder import Encoder, HiddenStateEncoder
from agentmem.readout import compose_prompt, format_memory_block


class MemorySession:
    """Dual-path memory: retrieve+store on pre, store-only on post."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        encoder: Encoder | None = None,
        bank: EpisodeBank | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.config.ensure_data_dir()

        self.encoder: Encoder = encoder or HiddenStateEncoder(
            model_id=self.config.encoder_model_id,
            layer=self.config.encoder_layer,
            device=self.config.encoder_device,
            dtype=self.config.encoder_dtype,
        )
        self.chunker = SemanticChunker(
            encoder=self.encoder,
            tau_break=self.config.tau_break,
        )
        self.bank = bank or EpisodeBank(
            dim=self.encoder.dim,
            data_dir=self.config.data_dir,
            tau_upsert=self.config.tau_upsert,
        )

    def pre(self, user_text: str) -> str:
        query_keys = self.chunker.query_keys(user_text)
        hits = self.bank.search(
            query_keys,
            top_k=self.config.retrieve_top_k,
            threshold=self.config.tau_retrieve,
        )
        memory_block = format_memory_block(
            hits, max_chars=self.config.max_readout_chars
        )

        for episode in self.chunker.chunk(user_text, role="user"):
            self.bank.upsert(episode)

        return compose_prompt(memory_block, user_text)

    def post(self, assistant_text: str) -> None:
        for episode in self.chunker.chunk(assistant_text, role="assistant"):
            self.bank.upsert(episode)
