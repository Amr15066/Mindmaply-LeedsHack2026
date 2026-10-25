"""
Schema migrations for the persisted mindmap document.

Every saved document carries ``schema_version``. ``migrate`` upgrades an older
document, one version at a time, to ``SCHEMA_VERSION``.

History:
    0  a single bare ConversationGraph (before multi-graph Mindmaps)
    1  Mindmap with "graphs"; no schema_version field
    2  schema_version added; ConversationMessage.embedding removed (never populated)
"""

import uuid
from typing import Any, Callable, Dict

SCHEMA_VERSION = 2


class UnrecognisedDocument(ValueError):
    """The JSON is not a mindmap document of any known version."""


class NewerSchemaError(ValueError):
    """The document was written by a newer version of the app."""


def detect_version(data: Dict[str, Any]) -> int:
    if "schema_version" in data:
        return int(data["schema_version"])
    if "graphs" in data:
        return 1
    if "blocks" in data:
        return 0
    raise UnrecognisedDocument("No 'schema_version', 'graphs' or 'blocks' key")


def _v0_to_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a bare graph in a Mindmap."""
    graph_id = data.get("graph_id") or str(uuid.uuid4())
    graph = dict(data, graph_id=graph_id)
    return {"graphs": {graph_id: graph}, "current_graph_id": graph_id, "metadata": {}}


def _v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the unused per-message embedding field."""
    for graph in data.get("graphs", {}).values():
        for message in graph.get("messages", {}).values():
            message.pop("embedding", None)
    return data


_STEPS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    0: _v0_to_v1,
    1: _v1_to_v2,
}


def migrate(data: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade ``data`` to the current schema (mutates and returns it)."""
    version = detect_version(data)
    if version > SCHEMA_VERSION:
        raise NewerSchemaError(
            f"Document schema v{version} is newer than this app (v{SCHEMA_VERSION})"
        )
    while version < SCHEMA_VERSION:
        data = _STEPS[version](data)
        version += 1
    data["schema_version"] = SCHEMA_VERSION
    return data
