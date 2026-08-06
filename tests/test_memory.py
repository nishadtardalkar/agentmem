"""Unit tests for AgentMem with a mocked HashEncoder (no GPU / HF download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agentmem.bank import EpisodeBank, MemoryEntry, RetrievedBucket
from agentmem.chunker import Episode, SemanticChunker, split_sentences
from agentmem.config import MemoryConfig
from agentmem.encoder import HashEncoder, l2_normalize
from agentmem.keys import HeuristicKeyExtractor, heuristic_keys, _parse_key_list
from agentmem.readout import compose_messages, compose_prompt, format_memory_block
from agentmem.session import MemorySession


@pytest.fixture
def encoder() -> HashEncoder:
    return HashEncoder(dim=32)


@pytest.fixture
def key_extractor() -> HeuristicKeyExtractor:
    return HeuristicKeyExtractor(max_keys=8)


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


def test_heuristic_keys_drops_stopwords():
    keys = heuristic_keys("I live in Boston.", max_keys=8)
    assert "live" in keys
    assert "boston" in keys
    assert "i" not in keys
    assert "in" not in keys


def test_parse_key_list_from_prose():
    raw = 'Sure: ["boston", "live"]'
    assert _parse_key_list(raw, max_keys=8) == ["boston", "live"]


def test_chunker_single_sentence(encoder: HashEncoder, key_extractor: HeuristicKeyExtractor):
    chunker = SemanticChunker(
        encoder=encoder, key_extractor=key_extractor, tau_break=0.99
    )
    eps = chunker.chunk("Hello world.", role="user")
    assert len(eps) == 1
    assert eps[0].keys.ndim == 2
    assert eps[0].keys.shape[1] == encoder.dim
    assert eps[0].keys.shape[0] >= 1


def test_chunker_breaks_on_low_similarity(
    encoder: HashEncoder, key_extractor: HeuristicKeyExtractor
):
    # Force break by using tau_break above any realistic sentence cosine
    chunker = SemanticChunker(
        encoder=encoder, key_extractor=key_extractor, tau_break=1.01
    )
    text = "Alpha sentence one. Completely different beta topic here."
    eps = chunker.chunk(text, role="user")
    assert len(eps) == 2
    assert all(ep.keys.ndim == 2 for ep in eps)



def test_bank_insert_and_retrieve(tmp_bank: EpisodeBank, encoder: HashEncoder):
    keys = encoder.encode_many(["boston", "live"])
    ep = Episode(
        role="user",
        text="I live in Boston.",
        sentences=["I live in Boston."],
        keys=keys,
    )
    lid = tmp_bank.upsert(ep)
    assert lid
    assert tmp_bank.ntotal == 2

    hits = tmp_bank.search(keys, top_k=3, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].latent_id == lid
    assert len(hits[0].entries) == 1
    assert hits[0].entries[0].text == "I live in Boston."
    assert hits[0].entries[0].ts == ep.timestamp


def test_bank_multi_key_same_latent(tmp_bank: EpisodeBank, encoder: HashEncoder):
    tokens = ["boston", "live", "dog", "rex"]
    keys = encoder.encode_many(tokens)
    ep = Episode(
        role="user",
        text="I live in Boston. I have a dog named Rex.",
        sentences=["I live in Boston.", "I have a dog named Rex."],
        keys=keys,
    )
    lid = tmp_bank.upsert(ep)
    assert tmp_bank.ntotal == 4

    # Query with only the dog/rex keys still returns the full bucket
    hits = tmp_bank.search(keys[2:4], top_k=3, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].latent_id == lid
    assert "Boston" in hits[0].entries[0].text and "Rex" in hits[0].entries[0].text


def test_bank_append_on_high_similarity(tmp_path: Path, encoder: HashEncoder):
    # Any cosine (including negative) triggers append when threshold is -1.
    bank = EpisodeBank(dim=encoder.dim, data_dir=tmp_path / "append", tau_upsert=-1.0)
    z1 = encoder.encode_many(["cats", "soft"])
    e1 = Episode(
        role="user",
        text="Cats are soft.",
        sentences=["Cats are soft."],
        keys=z1,
    )
    id1 = bank.upsert(e1)

    z2 = encoder.encode_many(["cats", "purr"])
    e2 = Episode(
        role="user",
        text="Cats purr loudly.",
        sentences=["Cats purr loudly."],
        keys=z2,
    )
    id2 = bank.upsert(e2)
    assert id1 == id2
    bucket = bank.get_latent(id1)
    assert bucket is not None
    assert len(bucket.entries) == 2
    assert bucket.entries[0].text == "Cats are soft."
    assert bucket.entries[0].ts == e1.timestamp
    assert bucket.entries[1].text == "Cats purr loudly."
    assert bucket.entries[1].ts == e2.timestamp
    assert bank.ntotal == 4
    assert bank.latent_count() == 1


def test_readout_compose_messages():
    buckets = [
        RetrievedBucket(
            latent_id="1",
            entries=[
                MemoryEntry(
                    role="user",
                    text="I live in Boston.",
                    sentences=["I live in Boston."],
                    ts="t",
                )
            ],
            score=0.9,
        )
    ]
    block = format_memory_block(buckets)
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
    encoder: HashEncoder, key_extractor: HeuristicKeyExtractor, tmp_path: Path
):
    """Shared subject tokens should retrieve across statement vs question form."""
    chunker = SemanticChunker(
        encoder=encoder, key_extractor=key_extractor, tau_break=0.99
    )
    bank = EpisodeBank(dim=encoder.dim, data_dir=tmp_path / "overlap", tau_upsert=0.99)

    store_eps = chunker.chunk("I live in Boston.", role="user")
    assert len(store_eps) == 1
    lid = bank.upsert(store_eps[0])

    query = chunker.query_keys("Where do I live?")
    assert query.shape[0] >= 1
    hits = bank.search(query, top_k=3, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].latent_id == lid
    assert "Boston" in hits[0].entries[0].text


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
    assert out1[-1]["role"] == "user"
    assert out1[-1]["content"] == "I live in Boston."
    assert out1[0]["role"] == "system"
    session.post("Nice city.")

    # Second turn: shared token keys should retrieve prior episode
    out2 = session.pre("Where do I live?")
    assert out2[-1]["content"] == "Where do I live?"
    assert "Boston" in out2[0]["content"]
    assert bank.ntotal >= 1
