import pytest
from helpers import CLASSIFY, blend, unit_vector

from mindmap_chat.config import config
from mindmap_chat.core.intent_detector import detect_intent_shift
from mindmap_chat.models import Block

INTENT_VEC = unit_vector("block intent")
OTHER_VEC = unit_vector("something else")


def block_with_intent():
    return Block(title="Topic", intent="block intent", embedding=list(INTENT_VEC))


def set_similarity(fake_llm, message, cos):
    fake_llm.embeddings[message] = blend(INTENT_VEC, OTHER_VEC, cos)


def test_high_similarity_continues_without_llm(fake_llm):
    set_similarity(fake_llm, "msg", config.thresholds.continue_threshold + 0.05)
    result = detect_intent_shift(fake_llm, block_with_intent(), "msg", [])
    assert result.action == "continue"
    assert result.confidence == pytest.approx(config.thresholds.continue_threshold + 0.05)
    assert fake_llm.prompts_of(CLASSIFY) == []


def test_medium_similarity_deepens_without_llm(fake_llm):
    set_similarity(fake_llm, "msg", config.thresholds.deepen_threshold + 0.05)
    result = detect_intent_shift(fake_llm, block_with_intent(), "msg", [])
    assert result.action == "deepen"
    assert fake_llm.prompts_of(CLASSIFY) == []


@pytest.mark.parametrize(
    "llm_label, expected",
    [("CONTINUE", "continue"), ("DEEPEN", "deepen"), ("NEW_CHILD", "new_child"),
     ("TANGENT", "tangent"), ("nonsense", "continue")],
)
def test_low_similarity_defers_to_llm(fake_llm, llm_label, expected):
    set_similarity(fake_llm, "msg", 0.1)
    fake_llm.classification = {
        "classification": llm_label,
        "confidence": 0.8,
        "reasoning": "because",
        "new_blocks": [{"title": "T", "intent": "I"}, "junk", {}],
    }
    result = detect_intent_shift(fake_llm, block_with_intent(), "msg", [])
    assert result.action == expected
    assert result.confidence == 0.8
    assert result.new_blocks == [{"title": "T", "intent": "I"}]
    assert len(fake_llm.prompts_of(CLASSIFY)) == 1


def test_legacy_single_block_fields_are_accepted(fake_llm):
    set_similarity(fake_llm, "msg", 0.1)
    fake_llm.classification = {
        "classification": "NEW_CHILD",
        "confidence": 0.7,
        "new_block_title": "Legacy",
    }
    result = detect_intent_shift(fake_llm, block_with_intent(), "msg", [])
    assert result.new_blocks == [{"title": "Legacy", "intent": "New discussion"}]


def test_llm_failure_falls_back_to_continue(fake_llm):
    set_similarity(fake_llm, "msg", 0.1)
    fake_llm.raw_responses = ["not json", "still not json"]
    result = detect_intent_shift(fake_llm, block_with_intent(), "msg", [])
    assert result.action == "continue"
    assert "Fallback" in result.reasoning
