import pytest
from helpers import add_message, make_tree

from mindmap_chat.models import Block, ConversationGraph, Mindmap


class TestAddBlock:
    def test_first_parentless_block_becomes_root(self):
        graph = ConversationGraph()
        root = Block()
        graph.add_block(root)
        assert graph.root_block_id == root.block_id

    def test_child_is_registered_on_parent(self):
        graph, root, a, *_ = make_tree()
        assert a.block_id in root.children

    def test_second_root_is_rejected(self):
        graph, *_ = make_tree()
        with pytest.raises(ValueError, match="Root block already set"):
            graph.add_block(Block())

    def test_missing_parent_is_rejected(self):
        graph = ConversationGraph()
        with pytest.raises(ValueError, match="Parent block"):
            graph.add_block(Block(parent_block_id="nope"))


def test_collect_descendants():
    graph, root, a, a1, a2, b = make_tree()
    assert set(graph.collect_descendants(root.block_id)) == {
        a.block_id, a1.block_id, a2.block_id, b.block_id
    }
    assert set(graph.collect_descendants(a.block_id)) == {a1.block_id, a2.block_id}
    assert graph.collect_descendants(b.block_id) == []
    assert graph.collect_descendants("unknown") == []


def test_delete_blocks_removes_their_messages():
    graph, root, a, a1, *_ = make_tree()
    keep = add_message(graph, root)
    gone = add_message(graph, a1)
    graph.delete_blocks([a1.block_id])
    assert a1.block_id not in graph.blocks
    assert gone.message_id not in graph.messages
    assert keep.message_id in graph.messages


def test_get_block_messages_preserves_order_and_skips_missing():
    graph, root, *_ = make_tree()
    m1 = add_message(graph, root, content="one")
    m2 = add_message(graph, root, role="assistant", content="two")
    root.conversation_refs.append("dangling")
    assert [m.message_id for m in graph.get_block_messages(root.block_id)] == [
        m1.message_id, m2.message_id
    ]
    assert graph.get_block_messages("unknown") == []


def test_rebuild_children_rederives_from_parent_ids():
    graph, root, a, a1, a2, b = make_tree()
    for block in graph.blocks.values():
        block.children = ["garbage"]
    graph.rebuild_children()
    assert sorted(root.children) == sorted([a.block_id, b.block_id])
    assert sorted(a.children) == sorted([a1.block_id, a2.block_id])
    assert a1.children == []


def test_graph_round_trip():
    graph, root, a, *_ = make_tree()
    add_message(graph, a, content="hello")
    graph.current_block_id = a.block_id
    clone = ConversationGraph.from_dict(graph.to_dict())
    assert clone.to_dict() == graph.to_dict()


def test_mindmap_round_trip_and_dangling_current_graph():
    mindmap = Mindmap()
    graph, *_ = make_tree()
    mindmap.add_graph(graph)
    assert Mindmap.from_dict(mindmap.to_dict()).to_dict() == mindmap.to_dict()

    data = mindmap.to_dict()
    data["current_graph_id"] = "deleted-graph"
    assert Mindmap.from_dict(data).current_graph_id == ""


def test_to_d3_graph_has_one_node_per_block_and_one_link_per_edge():
    graph, root, a, a1, a2, b = make_tree()
    graph.current_block_id = a.block_id
    d3 = graph.to_d3_graph()
    assert {n["id"] for n in d3["nodes"]} == set(graph.blocks)
    assert len(d3["links"]) == len(graph.blocks) - 1
    assert {(link["source"], link["target"]) for link in d3["links"]} == {
        (root.block_id, a.block_id),
        (a.block_id, a1.block_id),
        (a.block_id, a2.block_id),
        (root.block_id, b.block_id),
    }
    current = [n for n in d3["nodes"] if n["is_current"]]
    assert [n["id"] for n in current] == [a.block_id]


def test_d3_links_use_the_recorded_relation():
    """Regression (§2.3): every link was hard-coded to relation "child", confidence 0.8."""
    graph, root, a, a1, a2, b = make_tree()
    a.relation, a.relation_confidence, a.relation_reasoning = "deepen", 0.77, "went deeper"
    b.relation = "not-a-relation"
    links = {link["target"]: link for link in graph.to_d3_graph()["links"]}
    assert links[a.block_id]["relation"] == "deepen"
    assert links[a.block_id]["confidence"] == 0.77
    assert links[a.block_id]["reasoning"] == "went deeper"
    assert links[a.block_id]["color"] == "#66BB6A"
    assert links[b.block_id]["relation"] == "child"  # unknown values fall back


def test_blocks_saved_before_relations_existed_still_load():
    graph, root, a, *_ = make_tree()
    data = graph.to_dict()
    for block in data["blocks"].values():
        for key in ("relation", "relation_confidence", "relation_reasoning"):
            del block[key]
    loaded = ConversationGraph.from_dict(data)
    assert loaded.blocks[a.block_id].relation == "child"
    assert loaded.blocks[a.block_id].relation_confidence == 0.0
