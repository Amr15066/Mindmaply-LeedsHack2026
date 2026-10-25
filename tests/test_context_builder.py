from helpers import add_message, make_tree

from mindmap_chat.core.context_builder import (
    construct_block_context,
    construct_summary_prompt_context,
    format_ancestor_context,
)
from mindmap_chat.models import Block, ConversationGraph


def chain(depth):
    """root -> b1 -> b2 -> ... -> b{depth}; returns (graph, [root, b1, ...])."""
    graph = ConversationGraph()
    blocks = [Block(title="root", intent="root intent", summary="root summary")]
    graph.add_block(blocks[0])
    for i in range(1, depth + 1):
        block = Block(parent_block_id=blocks[-1].block_id, title=f"b{i}", intent=f"intent {i}")
        graph.add_block(block)
        blocks.append(block)
    return graph, blocks


def test_root_block_has_no_parent_section():
    graph, (root,) = chain(0)
    assert format_ancestor_context(graph, root) == ""
    assert "PARENT TOPICS" not in construct_block_context(graph, root)


def test_child_context_includes_parent_summary():
    """Regression (§2.8): a DEEPEN child lost the thread it was deepening from."""
    graph, (root, child) = chain(1)
    context = construct_block_context(graph, child)
    assert "PARENT TOPICS" in context
    assert "- root: root summary" in context
    assert context.index("PARENT TOPICS") < context.index("BLOCK CONTEXT")


def test_ancestors_are_nearest_first_and_use_intent_when_unsummarised():
    graph, blocks = chain(3)
    text = format_ancestor_context(graph, blocks[3], max_depth=10)
    assert text.index("- b2: intent 2") < text.index("- b1: intent 1") < text.index("- root:")


def test_depth_limit():
    graph, blocks = chain(5)
    text = format_ancestor_context(graph, blocks[5], max_depth=2)
    assert "- b4:" in text and "- b3:" in text
    assert "- b2:" not in text


def test_char_budget_stops_early_but_keeps_nearest_parent():
    graph, blocks = chain(3)
    blocks[2].summary = "x" * 500
    text = format_ancestor_context(graph, blocks[3], max_depth=10, max_chars=100)
    assert "- b2: xxx" in text
    assert text.rstrip().endswith("...")
    assert "- b1:" not in text


def test_cycle_in_parent_links_terminates():
    graph, (root, child) = chain(1)
    root.parent_block_id = child.block_id  # corrupt data
    text = format_ancestor_context(graph, child, max_depth=10)
    assert text.count("- root:") == 1


def test_summary_context_is_just_the_turns():
    graph, root, a, *_ = make_tree()
    add_message(graph, a, content="question")
    add_message(graph, a, role="assistant", content="answer")
    assert construct_summary_prompt_context(graph, a) == "User: question\n\nAssistant: answer"
