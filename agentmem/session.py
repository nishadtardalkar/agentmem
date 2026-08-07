from __future__ import annotations

from agentmem.bank import EpisodeBank, RetrievedEpisode
from agentmem.chunker import SemanticChunker
from agentmem.compress import EpisodeCompressor, ModelEpisodeCompressor
from agentmem.config import MemoryConfig
from agentmem.encoder import Encoder, HiddenStateEncoder
from agentmem.keys import KeyExtractor, ModelKeyExtractor
from agentmem.readout import compose_messages, format_memory_block


class MemorySession:
    """Dual-path memory: retrieve+store on pre, store-only on post."""

    def __init__(
        self,
        config: MemoryConfig | None = None,
        encoder: Encoder | None = None,
        bank: EpisodeBank | None = None,
        key_extractor: KeyExtractor | None = None,
        episode_compressor: EpisodeCompressor | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.config.ensure_data_dir()

        self.encoder: Encoder = encoder or HiddenStateEncoder(
            model_id=self.config.encoder_model_id,
            layer=self.config.encoder_layer,
            device=self.config.encoder_device,
            dtype=self.config.encoder_dtype,
        )
        if key_extractor is not None:
            self.key_extractor = key_extractor
        elif isinstance(self.encoder, HiddenStateEncoder):
            self.key_extractor = ModelKeyExtractor(self.encoder)
        else:
            raise TypeError(
                "key_extractor is required unless encoder is HiddenStateEncoder"
            )
        if episode_compressor is not None:
            self.episode_compressor = episode_compressor
        elif isinstance(self.encoder, HiddenStateEncoder):
            self.episode_compressor = ModelEpisodeCompressor(self.encoder)
        else:
            self.episode_compressor = None
        self.chunker = SemanticChunker(
            encoder=self.encoder,
            key_extractor=self.key_extractor,
            tau_break=self.config.tau_break,
            episode_compressor=self.episode_compressor,
        )
        self.bank = bank or EpisodeBank(
            dim=self.encoder.dim,
            data_dir=self.config.data_dir,
        )

    def pre(self, user_text: str) -> list[dict[str, str]]:
        """Retrieve memories, store the user half-turn, return chat messages."""
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
            self.bank.store(episode)

        return compose_messages(memory_block, user_text)

    def post(self, assistant_text: str) -> None:
        for episode in self.chunker.chunk(assistant_text, role="assistant"):
            self.bank.store(episode)

    def search(self, user_text: str) -> list[RetrievedEpisode]:
        """Retrieve related episodes without storing. Used by debug /search."""
        query_keys = self.chunker.query_keys(user_text)
        return self.bank.search(
            query_keys,
            top_k=self.config.retrieve_top_k,
            threshold=self.config.tau_retrieve,
        )
