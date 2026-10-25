"""FastAPI endpoints, with the offline FakeLLMClient and a temp storage file."""

import threading
import time

import pytest
from fastapi.testclient import TestClient
from helpers import blend, unit_vector

from mindmap_chat.config import config
from mindmap_chat.llm.base import LLMResponseError
from mindmap_chat.storage import JSONStorage


class CountingStorage(JSONStorage):
    def __init__(self, path):
        super().__init__(path)
        self.loads = 0
        self.saves = 0

    def load(self):
        self.loads += 1
        return super().load()

    def save(self, mindmap):
        self.saves += 1
        super().save(mindmap)

    def reset_counts(self):
        self.loads = self.saves = 0


@pytest.fixture
def store(tmp_path):
    return CountingStorage(str(tmp_path / "conv.json"))


@pytest.fixture
def client(store, fake_llm, monkeypatch):
    from app import app, routes

    monkeypatch.setattr(routes, "storage", store)
    monkeypatch.setattr(routes, "llm_client", fake_llm)
    with TestClient(app) as test_client:
        yield test_client


def new_mindmap(client, topic="Quantum computing"):
    response = client.post("/api/mindmaps/new", json={"topic": topic})
    assert response.status_code == 200
    return response.json()


def test_home_page_renders(client):
    """Regression: the name-first TemplateResponse call raised TypeError on Starlette 1.x."""
    new_mindmap(client)
    response = client.get("/")
    assert response.status_code == 200
    assert "Quantum computing" in response.text


def test_create_mindmap_uses_topic_as_title_with_one_load_and_save(client, store):
    """Regression (§2.11): create did load -> save -> load -> save."""
    body = new_mindmap(client)
    assert body["title"] == "Quantum computing"
    assert body["initial_response"] == "fake answer"
    assert (store.loads, store.saves) == (1, 1)

    listing = client.get("/api/mindmaps").json()["mindmaps"]
    assert [(m["title"], m["is_current"]) for m in listing] == [("Quantum computing", True)]


def test_chat_starts_then_continues_with_one_load_and_save_each(client, store, fake_llm):
    first = client.post("/api/chat", json={"content": "What is X?"}).json()
    assert first["assistant_response"] == "fake answer"
    assert [m["role"] for m in first["messages"]] == ["user", "assistant"]
    assert (store.loads, store.saves) == (1, 1)

    store.reset_counts()
    fake_llm.embeddings["more on X"] = unit_vector("fake intent")  # CONTINUE
    second = client.post("/api/chat", json={"content": "more on X"}).json()
    assert len(second["messages"]) == 4
    assert (store.loads, store.saves) == (1, 1)


def test_post_message_reports_the_block_it_landed_in(client, store, fake_llm):
    """Regression: after a DEEPEN, the old block's messages were returned."""
    created = new_mindmap(client)
    root_id = created["root_block_id"]
    sim = (config.thresholds.deepen_threshold + config.thresholds.continue_threshold) / 2
    fake_llm.embeddings["go deeper"] = blend(
        unit_vector("fake intent"), unit_vector("orthogonal"), sim
    )
    fake_llm.classification["new_blocks"] = [{"title": "Deeper", "intent": "deeper intent"}]

    store.reset_counts()
    body = client.post(f"/api/blocks/{root_id}/messages", json={"content": "go deeper"}).json()

    assert body["block_id"] == root_id
    assert body["current_block_id"] != root_id
    assert [m["content"] for m in body["messages"]] == ["go deeper", "fake answer"]
    assert body["assistant_response"] == "fake answer"
    assert (store.loads, store.saves) == (1, 1)  # was 2 loads + 2 saves


def test_unknown_block_is_404(client):
    assert client.post("/api/blocks/nope/messages", json={"content": "x"}).status_code == 404
    assert client.get("/api/blocks/nope/messages").status_code == 404
    assert client.post("/api/blocks/nope/switch").status_code == 404
    assert client.delete("/api/blocks/nope").status_code == 404


def test_get_graph_and_block_messages(client):
    created = new_mindmap(client)
    graph = client.get(f"/api/mindmaps/{created['graph_id']}/graph").json()
    assert [n["id"] for n in graph["nodes"]] == [created["root_block_id"]]
    block = client.get(f"/api/blocks/{created['root_block_id']}/messages").json()
    assert block["title"] == "Quantum computing"
    assert len(block["messages"]) == 2
    assert client.get("/api/mindmaps/nope/graph").status_code == 404


def test_delete_block_and_root_protection(client, fake_llm):
    created = new_mindmap(client)
    root_id = created["root_block_id"]
    fake_llm.classification = {
        "classification": "NEW_CHILD",
        "confidence": 0.8,
        "new_blocks": [{"title": "Child", "intent": "child intent"}],
    }
    chat = client.post("/api/chat", json={"content": "a subtopic"}).json()
    child_id = chat["current_block_id"]
    assert child_id != root_id

    body = client.delete(f"/api/blocks/{child_id}").json()
    assert body["current_block_id"] == root_id
    assert [n["id"] for n in body["graph"]["nodes"]] == [root_id]

    assert client.delete(f"/api/blocks/{root_id}").status_code == 400


def test_switch_and_delete_mindmaps(client, fake_llm):
    first = new_mindmap(client, "First")
    second = new_mindmap(client, "Second")

    assert client.post(f"/api/mindmaps/{first['graph_id']}/switch").json()["success"]
    listing = client.get("/api/mindmaps").json()["mindmaps"]
    assert {m["title"]: m["is_current"] for m in listing} == {"First": True, "Second": False}

    body = client.delete(f"/api/mindmaps/{first['graph_id']}").json()
    assert body["current_graph_id"] == second["graph_id"]
    assert client.post("/api/mindmaps/nope/switch").status_code == 404
    assert client.delete("/api/mindmaps/nope").status_code == 404


def test_empty_model_response_is_502_not_500(client, fake_llm):
    def blocked(prompt):
        raise LLMResponseError("Gemini returned no text (finish_reason=SAFETY)")

    fake_llm.answer = blocked
    response = client.post("/api/chat", json={"content": "something"})
    assert response.status_code == 502
    assert "SAFETY" in response.json()["detail"]


def test_concurrent_chats_do_not_lose_updates(client, fake_llm):
    """Two requests that overlap must both be persisted (read-modify-write under one lock)."""
    new_mindmap(client)
    for text in ("first concurrent", "second concurrent"):
        fake_llm.embeddings[text] = unit_vector("fake intent")  # CONTINUE

    def slow_answer(prompt):
        time.sleep(0.2)  # widen the race window
        return "fake answer"

    fake_llm.answer = slow_answer
    threads = [
        threading.Thread(target=client.post, args=("/api/chat",), kwargs={"json": {"content": t}})
        for t in ("first concurrent", "second concurrent")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    listing = client.get("/api/mindmaps").json()["mindmaps"]
    assert listing[0]["message_count"] == 6  # 2 initial + 2 per concurrent chat
