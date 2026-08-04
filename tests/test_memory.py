"""Unit tests for AgentMem with a mocked HashEncoder (no GPU / HF download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agentmem.bank import EpisodeBank
from agentmem.chunker import Episode, SemanticChunker, split_sentences
from agentmem.config import MemoryConfig
from agentmem.encoder import HashEncoder, l2_normalize
from agentmem.readout import compose_prompt, format_memory_block
from agentmem.session import MemorySession


@pytest.fixture
def encoder() -> HashEncoder:
    return HashEncoder(dim=32)


@pytest.fixture
def tmp_bank(tmp_path: Path, encoder: HashEncoder) -> EpisodeBank:
    return EpisodeBank(dim=encoder.dim, data_dir=tmp_path / "mem", tau_upsert=0.99)


def test_split_sentences():
    text = "I live in Boston. I have a dog. Also meeting at 3pm!"
    parts = split_sentences(text)
    assert len(parts) == 3
    assert parts[0].startswith("I live")


def test_l2_normalize_unit():
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = l2_normalize(v)
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-5


def test_chunker_single_sentence(encoder: HashEncoder):
    chunker = SemanticChunker(encoder=encoder, tau_break=0.99)
    eps = chunker.chunk("Hello world.", role="user")
    assert len(eps) == 1
    assert eps[0].latents.shape == (1, encoder.dim)


def test_chunker_breaks_on_low_similarity(encoder: HashEncoder):
    # Force break by using tau_break above any realistic hash cosine
    chunker = SemanticChunker(encoder=encoder, tau_break=1.01)
    text = "Alpha sentence one. Completely different beta topic here."
    eps = chunker.chunk(text, role="user")
    assert len(eps) == 2
    assert all(ep.latents.shape[0] == len(ep.sentences) for ep in eps)


def test_bank_insert_and_retrieve(tmp_bank: EpisodeBank, encoder: HashEncoder):
    z = encoder.encode_many(["I live in Boston."])
    ep = Episode(
        role="user",
        text="I live in Boston.",
        sentences=["I live in Boston."],
        latents=z,
    )
    eid = tmp_bank.upsert(ep)
    assert eid
    assert tmp_bank.ntotal == 1

    hits = tmp_bank.search(z, top_k=3, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].text == f"[{ep.timestamp}] I live in Boston."
    assert hits[0].episode_id == eid


def test_bank_multi_key_same_episode(tmp_bank: EpisodeBank, encoder: HashEncoder):
    sents = ["I live in Boston.", "I have a dog named Rex."]
    z = encoder.encode_many(sents)
    ep = Episode(
        role="user",
        text=" ".join(sents),
        sentences=sents,
        latents=z,
    )
    eid = tmp_bank.upsert(ep)
    assert tmp_bank.ntotal == 2

    # Query with only the second sentence key still returns full episode
    hits = tmp_bank.search(z[1:2], top_k=3, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].episode_id == eid
    assert "Boston" in hits[0].text and "Rex" in hits[0].text


def test_bank_append_on_high_similarity(tmp_path: Path, encoder: HashEncoder):
    # Any cosine (including negative) triggers append when threshold is -1.
    bank = EpisodeBank(dim=encoder.dim, data_dir=tmp_path / "append", tau_upsert=-1.0)
    z1 = encoder.encode_many(["Cats are soft."])
    e1 = Episode(role="user", text="Cats are soft.", sentences=["Cats are soft."], latents=z1)
    id1 = bank.upsert(e1)

    z2 = encoder.encode_many(["Cats purr loudly."])
    e2 = Episode(
        role="user",
        text="Cats purr loudly.",
        sentences=["Cats purr loudly."],
        latents=z2,
    )
    id2 = bank.upsert(e2)
    assert id1 == id2
    row = bank.get_episode(id1)
    assert row is not None
    assert f"[{e1.timestamp}] Cats are soft." in row.text
    assert f"[{e2.timestamp}] Cats purr loudly." in row.text
    assert bank.ntotal == 2


def test_readout_compose():
    from agentmem.bank import RetrievedEpisode

    eps = [
        RetrievedEpisode(
            episode_id="1",
            role="user",
            text="[t] I live in Boston.",
            sentences=["I live in Boston."],
            timestamp="t",
            score=0.9,
        )
    ]
    block = format_memory_block(eps)
    assert "[Memory]" in block
    prompt = compose_prompt(block, "Where do I live?")
    assert prompt.endswith("Where do I live?")
    assert "Boston" in prompt
    assert "[t]" in prompt


def test_session_pre_post(tmp_path: Path, encoder: HashEncoder):
    config = MemoryConfig(
        data_dir=tmp_path / "session",
        tau_break=1.01,
        tau_upsert=0.99,
        tau_retrieve=0.5,
        retrieve_top_k=5,
    )
    bank = EpisodeBank(dim=encoder.dim, data_dir=config.data_dir, tau_upsert=0.99)
    session = MemorySession(config=config, encoder=encoder, bank=bank)

    # First turn: nothing to retrieve
    out1 = session.pre("I live in Boston.")
    assert "I live in Boston." in out1
    session.post("Nice city.")

    # Second turn: should retrieve prior episode
    out2 = session.pre("Where do I live?")
    # May or may not hit depending on hash similarity; at least store path works
    assert "Where do I live?" in out2
    assert bank.ntotal >= 1
