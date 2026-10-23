import json

from helpers import add_message, make_tree

from mindmap_chat.models import Mindmap
from mindmap_chat.storage import JSONStorage


def test_missing_file_loads_empty_mindmap(tmp_path):
    mindmap = JSONStorage(str(tmp_path / "conv.json")).load()
    assert mindmap.graphs == {}


def test_round_trip(tmp_path):
    storage = JSONStorage(str(tmp_path / "conv.json"))
    mindmap = Mindmap()
    graph, root, a, *_ = make_tree()
    add_message(graph, a)
    mindmap.add_graph(graph)
    storage.save(mindmap)
    assert storage.load().to_dict() == mindmap.to_dict()


def test_load_rebuilds_children_from_parent_ids(tmp_path):
    path = tmp_path / "conv.json"
    storage = JSONStorage(str(path))
    mindmap = Mindmap()
    graph, root, a, *_ = make_tree()
    mindmap.add_graph(graph)
    storage.save(mindmap)

    data = json.loads(path.read_text())
    for block in data["graphs"][graph.graph_id]["blocks"].values():
        block["children"] = []
    path.write_text(json.dumps(data))

    loaded = storage.load().graphs[graph.graph_id]
    assert a.block_id in loaded.blocks[root.block_id].children


def test_legacy_single_graph_file_is_migrated(tmp_path):
    """Regression: the pre-Mindmap format (one bare graph) used to crash load()."""
    path = tmp_path / "conv.json"
    graph, root, a, *_ = make_tree()
    add_message(graph, a)
    data = graph.to_dict()
    for block in data["blocks"].values():
        block["children"] = []
    path.write_text(json.dumps(data))

    mindmap = JSONStorage(str(path)).load()
    assert list(mindmap.graphs) == [graph.graph_id]
    assert mindmap.current_graph_id == graph.graph_id
    loaded = mindmap.graphs[graph.graph_id]
    assert loaded.root_block_id == root.block_id
    assert a.block_id in loaded.blocks[root.block_id].children
    assert len(loaded.messages) == 1


def test_unrecognised_document_loads_empty_mindmap(tmp_path):
    """Regression: UnboundLocalError when the JSON has no "graphs" key."""
    path = tmp_path / "conv.json"
    path.write_text("{}")
    assert JSONStorage(str(path)).load().graphs == {}


def test_corrupted_file_loads_empty_mindmap(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text("{ not json")
    assert JSONStorage(str(path)).load().graphs == {}


def test_save_leaves_no_temp_files(tmp_path):
    storage = JSONStorage(str(tmp_path / "conv.json"))
    storage.save(Mindmap())
    assert [p.name for p in tmp_path.iterdir()] == ["conv.json"]


def test_clear_removes_file(tmp_path):
    path = tmp_path / "conv.json"
    storage = JSONStorage(str(path))
    storage.save(Mindmap())
    storage.clear()
    assert not path.exists()
