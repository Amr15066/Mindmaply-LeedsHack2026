import json

import pytest
from helpers import add_message, make_tree

from mindmap_chat.models import Mindmap
from mindmap_chat.storage import JSONStorage
from mindmap_chat.storage.migrations import SCHEMA_VERSION, NewerSchemaError, migrate


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


@pytest.mark.parametrize(
    "content",
    [
        "{}",  # regression: UnboundLocalError when the JSON has no "graphs" key
        "{ not json",
        "[1, 2, 3]",
    ],
)
def test_unreadable_file_is_kept_aside_and_mindmap_starts_empty(tmp_path, content):
    path = tmp_path / "conv.json"
    path.write_text(content)
    storage = JSONStorage(str(path))

    assert storage.load().graphs == {}
    (backup,) = [p for p in tmp_path.iterdir() if p.name.startswith("conv.json.unreadable-")]
    assert backup.read_text() == content

    storage.save(Mindmap())  # does not destroy the backup
    assert backup.exists()


def test_saved_file_records_schema_version(tmp_path):
    path = tmp_path / "conv.json"
    JSONStorage(str(path)).save(Mindmap())
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION


def test_v1_file_with_message_embeddings_is_migrated(tmp_path):
    """v1 files serialised an always-empty ConversationMessage.embedding field."""
    path = tmp_path / "conv.json"
    mindmap = Mindmap()
    graph, root, a, *_ = make_tree()
    msg = add_message(graph, a)
    mindmap.add_graph(graph)
    data = mindmap.to_dict()  # no schema_version -> v1
    data["graphs"][graph.graph_id]["messages"][msg.message_id]["embedding"] = []
    path.write_text(json.dumps(data))

    loaded = JSONStorage(str(path)).load()
    assert loaded.graphs[graph.graph_id].messages[msg.message_id].content == "hi"


def test_newer_schema_is_refused_not_overwritten(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "graphs": {}}))
    with pytest.raises(NewerSchemaError):
        JSONStorage(str(path)).load()
    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION + 1


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


def test_migrate_v0_chain_reaches_current_version():
    graph, *_ = make_tree()
    data = migrate(graph.to_dict())
    assert data["schema_version"] == SCHEMA_VERSION
    assert list(data["graphs"]) == [graph.graph_id]
    assert Mindmap.from_dict(data).current_graph_id == graph.graph_id


def test_migrate_is_idempotent_on_current_documents():
    mindmap = Mindmap()
    mindmap.add_graph(make_tree()[0])
    data = {"schema_version": SCHEMA_VERSION, **mindmap.to_dict()}
    assert migrate(json.loads(json.dumps(data))) == data
