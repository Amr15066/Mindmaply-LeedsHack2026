import pytest
from helpers import ANSWER, SUMMARY, blend, unit_vector

from mindmap_chat.config import config
from mindmap_chat.conversation import ConversationManager
from mindmap_chat.models import Block
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


def test_active_graph_requires_a_conversation(manager):
    with pytest.raises(ValueError, match="No active graph"):
        manager.active_graph()
    manager.start_new_conversation("What is X?")
    assert manager.active_graph() is manager.graph


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


# --- re-summarisation (§2.7) ------------------------------------------------

def chat_in_same_block(manager, fake_llm, n_turns):
    root = manager.graph.blocks[manager.graph.root_block_id]
    for i in range(n_turns):
        message = f"follow-up {i}"
        fake_llm.embeddings[message] = list(root.embedding)  # always CONTINUE
        manager.continue_conversation(message)
    return root


def test_block_is_resummarised_every_n_messages(manager, fake_llm, monkeypatch):
    """Regression (§2.7): a block was summarised once, at N messages, then never again."""
    monkeypatch.setattr(config, "auto_summarize_after_n_messages", 6)
    manager.start_new_conversation("What is X?")  # 2 messages

    root = chat_in_same_block(manager, fake_llm, 2)  # 6 messages
    assert len(fake_llm.prompts_of(SUMMARY)) == 1
    assert root.last_summarized_at == 6

    chat_in_same_block(manager, fake_llm, 2)  # 10 messages: only 4 new
    assert len(fake_llm.prompts_of(SUMMARY)) == 1

    fake_llm.summary["summary"] = "updated summary"
    chat_in_same_block(manager, fake_llm, 1)  # 12 messages: 6 new
    assert len(fake_llm.prompts_of(SUMMARY)) == 2
    assert root.summary == "updated summary"
    assert root.last_summarized_at == 12


def test_failed_summary_is_retried_next_turn(manager, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "auto_summarize_after_n_messages", 4)
    manager.start_new_conversation("What is X?")
    root = manager.graph.blocks[manager.graph.root_block_id]

    good_summary = fake_llm.summary
    fake_llm.summary = None  # json.dumps(None) -> "null" -> .get() fails
    chat_in_same_block(manager, fake_llm, 1)  # 4 messages -> attempt fails
    assert root.last_summarized_at == 0

    fake_llm.summary = good_summary
    chat_in_same_block(manager, fake_llm, 1)  # 6 messages -> retried
    assert root.last_summarized_at == 6
    assert root.summary == "fake summary"


def test_summary_prompt_lists_intent_and_messages_once(manager, fake_llm, monkeypatch):
    monkeypatch.setattr(config, "auto_summarize_after_n_messages", 4)
    manager.start_new_conversation("What is X?")
    chat_in_same_block(manager, fake_llm, 1)
    (prompt,) = fake_llm.prompts_of(SUMMARY)
    assert prompt.count("BLOCK INTENT") == 1
    assert prompt.count("MESSAGES IN THIS BLOCK") == 1
    assert prompt.count("User: What is X?") == 1


# --- relation persistence (§2.3) -------------------------------------------

def start(manager):
    manager.start_new_conversation("What is X?")
    return manager.graph.blocks[manager.graph.root_block_id]


def test_deepen_records_relation_and_similarity(manager, fake_llm):
    root = start(manager)
    sim = (config.thresholds.deepen_threshold + config.thresholds.continue_threshold) / 2
    fake_llm.embeddings["How does X work inside?"] = blend(
        root.embedding, unit_vector("orthogonal"), sim
    )
    fake_llm.classification["new_blocks"] = [{"title": "X internals", "intent": "inner workings"}]

    manager.continue_conversation("How does X work inside?")

    child = manager.graph.blocks[manager.graph.current_block_id]
    assert child.parent_block_id == root.block_id
    assert child.relation == "deepen"
    assert child.relation_confidence == pytest.approx(sim)
    assert "deepens" in child.relation_reasoning


def test_new_child_records_classifier_output(manager, fake_llm):
    root = start(manager)
    fake_llm.classification = {
        "classification": "NEW_CHILD",
        "confidence": 0.83,
        "reasoning": "related subtopic",
        "new_blocks": [{"title": "Y", "intent": "about Y"}],
    }
    manager.continue_conversation("What about Y?")

    child = manager.graph.blocks[manager.graph.current_block_id]
    assert child.parent_block_id == root.block_id
    assert (child.relation, child.relation_confidence, child.relation_reasoning) == (
        "child", 0.83, "related subtopic"
    )


def test_tangent_matched_in_other_graph_records_sibling(manager, fake_llm):
    # graph A about X, then graph B about Z (current)
    root_a = start(manager)
    graph_a = manager.graph
    fake_llm.intent = {"intent": "z intent", "title": "Z"}
    manager.start_new_conversation("Tell me about Z")

    # A tangent from Z that is related (but not identical) to X's root block
    fake_llm.classification = {
        "classification": "TANGENT",
        "confidence": 0.9,
        "reasoning": "unrelated to Z",
        "new_blocks": [{"title": "X again", "intent": "x follow-up"}],
    }
    sim = (config.thresholds.sibling_threshold + config.thresholds.continue_threshold) / 2
    fake_llm.embeddings["back to X?"] = blend(root_a.embedding, unit_vector("orthogonal"), sim)

    manager.continue_conversation("back to X?")

    assert manager.graph is graph_a
    child = manager.graph.blocks[manager.graph.current_block_id]
    assert child.parent_block_id == root_a.block_id
    assert child.relation == "sibling"
    assert child.relation_confidence == pytest.approx(sim)
    assert "Fake Title" in child.relation_reasoning  # names the matched block


def test_child_is_registered_on_parent_exactly_once(manager, fake_llm):
    root = start(manager)
    fake_llm.classification = {
        "classification": "NEW_CHILD",
        "confidence": 0.8,
        "new_blocks": [{"title": "A", "intent": "a"}, {"title": "B", "intent": "b"}],
    }
    manager.continue_conversation("two subtopics")
    assert len(root.children) == len(set(root.children)) == 2


def test_delete_block_removes_subtree_and_moves_to_parent(manager, fake_llm):
    root = start(manager)
    fake_llm.classification = {
        "classification": "NEW_CHILD",
        "confidence": 0.8,
        "new_blocks": [{"title": "A", "intent": "a"}],
    }
    manager.continue_conversation("subtopic A")
    child_id = manager.graph.current_block_id
    fake_llm.classification["new_blocks"] = [{"title": "A1", "intent": "a1"}]
    manager.continue_conversation("deeper in A")
    grandchild_id = manager.graph.current_block_id

    manager.delete_block(child_id)

    assert set(manager.graph.blocks) == {root.block_id}
    assert root.children == []
    assert manager.graph.current_block_id == root.block_id
    assert {m.block_id for m in manager.graph.messages.values()} == {root.block_id}
    assert grandchild_id not in manager.graph.blocks
    with pytest.raises(ValueError, match="root"):
        manager.delete_block(root.block_id)


def test_tangent_turn_embeds_the_message_once(manager, fake_llm):
    """Regression (§2.9): the message was embedded again for the cross-graph search."""
    start(manager)
    fake_llm.classification = {"classification": "TANGENT", "confidence": 0.9}
    fake_llm.intent = {"intent": "cooking", "title": "Cooking"}

    manager.continue_conversation("How do I make bread?")

    assert fake_llm.embedded.count("How do I make bread?") == 1


def test_missing_block_embeddings_are_backfilled_in_one_batch(manager, fake_llm):
    """Regression (§3.4): blocks were embedded one network call at a time inside the scan."""
    start(manager)
    for i in range(3):
        manager.graph.add_block(Block(
            parent_block_id=manager.graph.root_block_id, title=f"b{i}", intent=f"intent {i}"
        ))
    fake_llm.classification = {"classification": "TANGENT", "confidence": 0.9}

    manager.continue_conversation("Unrelated question")

    assert fake_llm.batches == [["intent 0", "intent 1", "intent 2"]]
    blocks = [b for b in manager.mindmap.graphs[manager.graph.graph_id].blocks.values()]
    assert all(b.embedding for b in blocks)


def test_embeddings_of_another_dimension_are_ignored(manager, fake_llm):
    root = start(manager)
    root.embedding = [1.0, 0.0, 0.0]  # e.g. from a different embedding model
    fake_llm.classification = {"classification": "TANGENT", "confidence": 0.9}
    fake_llm.intent = {"intent": "cooking", "title": "Cooking"}

    manager.continue_conversation("Unrelated question")  # must not crash

    assert manager.graph.blocks[manager.graph.root_block_id].title == "Cooking"


def test_relations_survive_storage_and_reach_d3(manager, fake_llm, storage):
    start(manager)
    fake_llm.classification = {
        "classification": "NEW_CHILD",
        "confidence": 0.83,
        "reasoning": "related subtopic",
        "new_blocks": [{"title": "Y", "intent": "about Y"}],
    }
    manager.continue_conversation("What about Y?")

    graph = storage.load().graphs[manager.graph.graph_id]
    (link,) = graph.to_d3_graph()["links"]
    assert link["relation"] == "child"
    assert link["confidence"] == 0.83
    assert link["reasoning"] == "related subtopic"
    assert link["color"] == "#2196F3"
