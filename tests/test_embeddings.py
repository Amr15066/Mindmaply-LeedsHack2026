import numpy as np
import pytest

from mindmap_chat.core.embeddings import compute_similarity, cosine_batch, embed_text
from mindmap_chat.llm.base import LLMClient


def test_identical_vectors_have_similarity_one():
    assert compute_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_scale_invariant():
    assert compute_similarity([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert compute_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one():
    assert compute_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


@pytest.mark.parametrize("a, b", [([], [1.0]), ([1.0], []), ([], [])])
def test_empty_vectors_return_zero(a, b):
    assert compute_similarity(a, b) == 0.0


def test_zero_vector_returns_zero():
    assert compute_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_mismatched_dimensions_return_zero():
    """zip() used to truncate silently and return a meaningless score."""
    assert compute_similarity([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_batch_matches_pairwise():
    rng = np.random.default_rng(0)
    query = rng.normal(size=16)
    matrix = rng.normal(size=(5, 16))
    expected = [compute_similarity(query.tolist(), row.tolist()) for row in matrix]
    assert cosine_batch(query, matrix) == pytest.approx(expected)


def test_cosine_batch_zero_rows_and_empty_matrix():
    sims = cosine_batch([1.0, 0.0], np.array([[0.0, 0.0], [2.0, 0.0]]))
    assert sims.tolist() == [0.0, 1.0]
    assert cosine_batch([1.0, 0.0], np.empty((0, 2))).tolist() == []


def test_default_embed_batch_embeds_each_text(fake_llm):
    vectors = LLMClient.embed_batch(fake_llm, ["a", "b"])
    assert vectors == [fake_llm.embed("a"), fake_llm.embed("b")]


def test_embed_text_delegates_to_client(fake_llm):
    fake_llm.embeddings["hello"] = [0.5, 0.5]
    assert embed_text(fake_llm, "hello") == [0.5, 0.5]
    assert fake_llm.embedded == ["hello"]
