"""Unit tests for AgentMem with a mocked HashEncoder (no GPU / HF download)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from agentmem.bank import EpisodeBank, MemoryEntry, RetrievedEpisode
from agentmem.chunker import Episode, SemanticChunker, split_sentences
from agentmem.config import MemoryConfig
from agentmem.encoder import HashEncoder, l2_normalize
from agentmem.keys import _parse_key_list
from agentmem.readout import compose_messages, compose_prompt, format_memory_block
from agentmem.session import MemorySession


class StubKeyExtractor:
    """Deterministic keys for unit tests (no LLM)."""

    def extract(self, text: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for match in re.findall(r"[a-z0-9]+", text.lower()):
            if match in seen:
                continue
            seen.add(match)
            out.append(match)
        return out

    def extract_many(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.extract(t) for t in texts]

@pytest.fixture
def encoder() -> HashEncoder:
    return HashEncoder(dim=32)


@pytest.fixture
def key_extractor() -> StubKeyExtractor:
    return StubKeyExtractor()


@pytest.fixture
def tmp_bank(tmp_path: Path, encoder: HashEncoder) -> EpisodeBank:
    return EpisodeBank(dim=encoder.dim, data_dir=tmp_path / "mem")


def test_split_sentences():
    text = "I live in Boston. I have a dog. Also meeting at 3pm!"
    parts = split_sentences(text)
    assert len(parts) == 3
    assert parts[0].startswith("I live")


def test_l2_normalize_unit():
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = l2_normalize(v)
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-5


def test_parse_key_list_from_prose():
    raw = 'Sure: ["boston", "live"]'
    assert _parse_key_list(raw) == ["boston", "live"]


def test_chunker_single_sentence(encoder: HashEncoder, key_extractor: StubKeyExtractor):
    chunker = SemanticChunker(
        encoder=encoder, key_extractor=key_extractor, tau_break=0.99
    )
    eps = chunker.chunk("Hello world.", role="user")
    assert len(eps) == 1
    assert eps[0].keys.ndim == 2
    assert eps[0].keys.shape[1] == encoder.dim
    assert eps[0].keys.shape[0] >= 1


def test_chunker_breaks_on_low_similarity(
    encoder: HashEncoder, key_extractor: StubKeyExtractor
):
    # Force break by using tau_break above any realistic sentence cosine
    chunker = SemanticChunker(
        encoder=encoder, key_extractor=key_extractor, tau_break=1.01
    )
    text = "Alpha sentence one. Completely different beta topic here."
    eps = chunker.chunk(text, role="user")
    assert len(eps) == 2
    assert all(ep.keys.ndim == 2 for ep in eps)


def test_chunker_compresses_before_store(
    encoder: HashEncoder, key_extractor: StubKeyExtractor
):
    class FixedCompressor:
        def compress(self, text: str) -> str:
            return "Nishad is a coder. Sam is a singer. Tom is a pilot."

    chunker = SemanticChunker(
        encoder=encoder,
        key_extractor=key_extractor,
        tau_break=-1.0,  # keep one episode so compression is the focus
        episode_compressor=FixedCompressor(),
    )
    raw = (
        "I am Nishad. Im a coder. I have a friend sam who is a singer "
        "and a friend tom who is a pilot."
    )
    eps = chunker.chunk(raw, role="user")
    assert len(eps) == 1
    assert eps[0].text == "Nishad is a coder. Sam is a singer. Tom is a pilot."
    assert "nishad" in key_extractor.extract(eps[0].text)
    assert raw not in eps[0].text


def test_bank_insert_and_retrieve(tmp_bank: EpisodeBank, encoder: HashEncoder):
    keys = encoder.encode_many(["boston", "live"])
    ep = Episode(
        role="user",
        text="I live in Boston.",
        sentences=["I live in Boston."],
        keys=keys,
    )
    eid = tmp_bank.store(ep)
    assert eid
    assert tmp_bank.ntotal == 2

    hits = tmp_bank.search(keys, top_k=3, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].episode_id == eid
    assert hits[0].entry.text == "I live in Boston."
    assert hits[0].entry.ts == ep.timestamp


def test_bank_multi_key_same_episode(tmp_bank: EpisodeBank, encoder: HashEncoder):
    tokens = ["boston", "live", "dog", "rex"]
    keys = encoder.encode_many(tokens)
    ep = Episode(
        role="user",
        text="I live in Boston. I have a dog named Rex.",
        sentences=["I live in Boston.", "I have a dog named Rex."],
        keys=keys,
    )
    eid = tmp_bank.store(ep)
    assert tmp_bank.ntotal == 4

    # Query with only the dog/rex keys still returns the full episode
    hits = tmp_bank.search(keys[2:4], top_k=3, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].episode_id == eid
    assert "Boston" in hits[0].entry.text and "Rex" in hits[0].entry.text


def test_bank_episodes_never_merge(tmp_path: Path, encoder: HashEncoder):
    """Each episode is its own value; similar keys do not collapse stores."""
    bank = EpisodeBank(dim=encoder.dim, data_dir=tmp_path / "nomerge")
    z1 = encoder.encode_many(["cats", "soft"])
    e1 = Episode(
        role="user",
        text="Cats are soft.",
        sentences=["Cats are soft."],
        keys=z1,
    )
    id1 = bank.store(e1)

    # Same subject token ("cats") must still create a distinct episode.
    z2 = encoder.encode_many(["cats", "purr"])
    e2 = Episode(
        role="user",
        text="Cats purr loudly.",
        sentences=["Cats purr loudly."],
        keys=z2,
    )
    id2 = bank.store(e2)
    assert id1 != id2
    assert bank.episode_count() == 2
    assert bank.ntotal == 4

    ep1 = bank.get_episode(id1)
    ep2 = bank.get_episode(id2)
    assert ep1 is not None and ep1.entry.text == "Cats are soft."
    assert ep2 is not None and ep2.entry.text == "Cats purr loudly."

    # Retrieving on "cats" can surface both episodes.
    hits = bank.search(encoder.encode_many(["cats"]), top_k=5, threshold=0.5)
    hit_ids = {h.episode_id for h in hits}
    assert id1 in hit_ids and id2 in hit_ids


def test_readout_compose_messages():
    episodes = [
        RetrievedEpisode(
            episode_id="1",
            entry=MemoryEntry(
                role="user",
                text="I live in Boston.",
                sentences=["I live in Boston."],
                ts="t",
            ),
            score=0.9,
        )
    ]
    block = format_memory_block(episodes)
    assert "[Memory]" in block
    assert "Boston" in block
    messages = compose_messages(block, "Where do I live?")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "NOT the user's current message" in messages[0]["content"]
    assert "Boston" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Where do I live?"}
    prompt = compose_prompt(block, "Where do I live?")
    assert "[system]" in prompt and "[user]" in prompt


def test_query_keys_overlap_store_keys(
    encoder: HashEncoder, key_extractor: StubKeyExtractor, tmp_path: Path
):
    """Shared subject tokens should retrieve across statement vs question form."""
    chunker = SemanticChunker(
        encoder=encoder, key_extractor=key_extractor, tau_break=0.99
    )
    bank = EpisodeBank(dim=encoder.dim, data_dir=tmp_path / "overlap")

    store_eps = chunker.chunk("I live in Boston.", role="user")
    assert len(store_eps) == 1
    eid = bank.store(store_eps[0])

    query = chunker.query_keys("Where do I live?")
    assert query.shape[0] >= 1
    hits = bank.search(query, top_k=3, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].episode_id == eid
    assert "Boston" in hits[0].entry.text


def test_session_pre_post(
    tmp_path: Path, encoder: HashEncoder, key_extractor: StubKeyExtractor
):
    config = MemoryConfig(
        data_dir=tmp_path / "session",
        tau_break=1.01,
        tau_retrieve=0.5,
        retrieve_top_k=5,
    )
    bank = EpisodeBank(dim=encoder.dim, data_dir=config.data_dir)
    session = MemorySession(
        config=config, encoder=encoder, bank=bank, key_extractor=key_extractor
    )

    # First turn: nothing to retrieve
    out1 = session.pre("I live in Boston.")
    assert out1[-1]["role"] == "user"
    assert out1[-1]["content"] == "I live in Boston."
    assert out1[0]["role"] == "system"
    session.post("Nice city.")

    # Second turn: shared token keys should retrieve prior episode
    out2 = session.pre("Where do I live?")
    assert out2[-1]["content"] == "Where do I live?"
    assert "Boston" in out2[0]["content"]
    assert bank.ntotal >= 1
    assert bank.episode_count() >= 2  # user + assistant already stored
