import pytest

from mindmap_chat.core.embeddings import compute_similarity, embed_text


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


def test_embed_text_delegates_to_client(fake_llm):
    fake_llm.embeddings["hello"] = [0.5, 0.5]
    assert embed_text(fake_llm, "hello") == [0.5, 0.5]
    assert fake_llm.embedded == ["hello"]
