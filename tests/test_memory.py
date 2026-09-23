"""Tests for the memforensics memory store and retriever.

Split in two: the search/ablation logic is tested against a stub embedder
(fast, no model download), and a small set of checks pin the real
MiniLM embedder's determinism — the property the whole counterfactual
method rests on.
"""
from __future__ import annotations

import numpy as np
import pytest

from memforensics.memory import MemoryState, MemoryWrite, MiniLMEmbedder, build_store

DIM = 384


class StubEmbedder:
    """Maps each distinct string to a fixed one-hot-ish unit vector."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), DIM))
        for i, text in enumerate(texts):
            slot = self._index.setdefault(text, len(self._index))
            rows[i, slot % DIM] = 1.0
        return rows


def w(session_id: int, seq: int, content: str) -> MemoryWrite:
    return MemoryWrite(session_id=session_id, seq=seq, content=content)


# --------------------------------------------------------------------- #
# MemoryWrite
# --------------------------------------------------------------------- #
def test_id_is_derived_from_session_and_seq():
    assert w(3, 7, "anything").id == "3-7"


def test_write_is_frozen_and_hashable():
    write = w(1, 0, "content")
    with pytest.raises(Exception):
        write.content = "mutated"  # type: ignore[misc]
    assert {write, w(1, 0, "content")} == {write}


def test_content_is_stored_verbatim():
    raw = "  Mixed Case\nwith newline and trailing space  "
    assert w(1, 0, raw).content == raw


# --------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------- #
def test_empty_store_returns_nothing():
    store = MemoryState(StubEmbedder())
    assert store.search("anything", top_k=5) == []


def test_top_k_is_clipped_to_store_size():
    store = build_store([w(1, 0, "a"), w(1, 1, "b")], StubEmbedder())
    assert len(store.search("a", top_k=10)) == 2


def test_search_orders_by_descending_similarity():
    embedder = StubEmbedder()
    target = w(1, 1, "target")
    store = build_store([w(1, 0, "other"), target, w(1, 2, "another")], embedder)
    assert store.search("target", top_k=3)[0] is target


def test_ties_are_broken_by_insertion_order():
    first, second = w(1, 0, "same"), w(2, 0, "same")
    store = build_store([first, second], StubEmbedder())
    assert [r.id for r in store.search("same", top_k=2)] == [first.id, second.id]


# --------------------------------------------------------------------- #
# Ablation: build_store restricts the candidate pool
# --------------------------------------------------------------------- #
def test_build_store_only_searches_the_given_subset():
    embedder = StubEmbedder()
    poisoned = w(2, 0, "poisoned")
    all_writes = [w(1, 0, "benign"), poisoned]

    assert poisoned in build_store(all_writes, embedder).search("poisoned", top_k=2)

    ablated = build_store([x for x in all_writes if x is not poisoned], embedder)
    assert poisoned not in ablated.search("poisoned", top_k=2)


def test_batched_build_matches_incremental_add():
    writes = [w(1, i, f"content {i}") for i in range(4)]

    incremental = MemoryState(StubEmbedder())
    for write in writes:
        incremental.add(write)

    assert np.allclose(build_store(writes, StubEmbedder()).vectors, incremental.vectors)


# --------------------------------------------------------------------- #
# The real embedder
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def embedder() -> MiniLMEmbedder:
    try:
        return MiniLMEmbedder()
    except Exception as exc:  # offline, or weights unavailable
        pytest.skip(f"MiniLM unavailable: {exc}")


def test_embeddings_are_unit_length_and_384_dim(embedder: MiniLMEmbedder):
    vectors = embedder.embed(["first sentence", "second sentence"])
    assert vectors.shape == (2, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_same_input_embeds_identically_within_a_process(embedder: MiniLMEmbedder):
    texts = ["Alice's email is alice@example.com", "the weekly summary goes to the team"]
    assert np.array_equal(embedder.embed(texts), embedder.embed(texts))


def test_same_input_embeds_identically_across_embedder_instances(embedder: MiniLMEmbedder):
    texts = ["Alice's email is alice@example.com"]
    assert np.allclose(embedder.embed(texts), MiniLMEmbedder().embed(texts), atol=1e-6)


def test_retrieval_is_semantic_not_lexical(embedder: MiniLMEmbedder):
    contact = w(1, 0, "Alice's email address is alice@example.com")
    store = build_store(
        [contact, w(1, 1, "The team standup is at 9am on Mondays")],
        embedder,
    )
    assert store.search("how do I reach Alice?", top_k=1) == [contact]
