import pytest
from helpers import ANSWER, unit_vector

from mindmap_chat.conversation import ConversationManager
from mindmap_chat.storage import JSONStorage


@pytest.fixture
def storage(tmp_path):
    return JSONStorage(str(tmp_path / "conv.json"))


@pytest.fixture
def manager(fake_llm, storage):
    return ConversationManager(fake_llm, storage)


def same_topic(fake_llm, message, intent="fake intent"):
    """Make ``message`` embed identically to ``intent`` so the turn is a CONTINUE."""
    fake_llm.embeddings[message] = unit_vector(intent)


def test_start_new_conversation_creates_root_with_one_exchange(manager, fake_llm, storage):
    response = manager.start_new_conversation("What is X?")
    assert response == "fake answer"

    graph = manager.graph
    root = graph.blocks[graph.root_block_id]
    assert root.title == "Fake Title"
    assert root.intent == "fake intent"
    assert [(m.role, m.content) for m in graph.get_block_messages(root.block_id)] == [
        ("user", "What is X?"),
        ("assistant", "fake answer"),
    ]
    assert storage.load().to_dict() == manager.mindmap.to_dict()


def test_answer_prompt_has_context_and_new_message_exactly_once(manager, fake_llm):
    """Regression (§2.2): block context was rendered twice and the new message duplicated."""
    manager.start_new_conversation("What is X?")
    same_topic(fake_llm, "Tell me more about X")
    manager.continue_conversation("Tell me more about X")

    prompt = fake_llm.prompts_of(ANSWER)[-1]
    assert prompt.count("Tell me more about X") == 1
    assert prompt.count("Title: Fake Title") == 1
    assert prompt.count("KEY POINTS COVERED") == 1
    # previous exchange is in the history
    assert "User: What is X?" in prompt
    assert "Assistant: fake answer" in prompt


def test_first_answer_prompt_has_empty_history(manager, fake_llm):
    manager.start_new_conversation("What is X?")
    prompt = fake_llm.prompts_of(ANSWER)[0]
    assert prompt.count("What is X?") == 1
    assert "(No messages yet)" in prompt


def test_failed_answer_stores_nothing(manager, fake_llm, storage):
    manager.start_new_conversation("What is X?")
    same_topic(fake_llm, "boom")

    def explode(prompt):
        raise RuntimeError("LLM down")

    fake_llm.answer = explode
    with pytest.raises(RuntimeError):
        manager.continue_conversation("boom")
    assert all(m.content != "boom" for m in manager.graph.messages.values())
