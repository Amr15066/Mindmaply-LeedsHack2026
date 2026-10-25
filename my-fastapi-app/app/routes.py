from threading import Lock

from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from app import app, templates
from mindmap_chat.storage import JSONStorage
from mindmap_chat.conversation import ConversationManager
from mindmap_chat.llm.base import LLMResponseError
from mindmap_chat.llm.gemini import GeminiClient
from mindmap_chat.config import config, validate_config
from mindmap_chat.models import ConversationGraph, Mindmap

# Initialize backends (lazy - only validate when actually needed).
# Keep LLM and storage cached, but ALWAYS create a fresh ConversationManager
# so it works on the latest mindmap state from disk on every request.
storage = None
llm_client = None

# Handlers are plain `def`, so FastAPI runs them in a threadpool and blocking
# LLM/file calls no longer stall the event loop. Every request that modifies
# the mindmap holds this lock for its whole load -> mutate -> save, so
# concurrent requests in this process cannot overwrite each other's changes.
# (It does not protect across processes: run a single worker until storage
# moves to a database.)
write_lock = Lock()

def get_storage():
    """Lazy initialize storage."""
    global storage
    if storage is None:
        storage = JSONStorage(config.storage_path)
    return storage

def get_llm_client():
    """Lazy initialize LLM client."""
    global llm_client
    if llm_client is None:
        validate_config()  # Only validate when needed
        llm_client = GeminiClient()
    return llm_client

def get_conversation_manager(mindmap: Mindmap) -> ConversationManager:
    """Return a fresh ConversationManager bound to an already-loaded mindmap."""
    return ConversationManager(get_llm_client(), get_storage(), mindmap=mindmap)


def find_block_graph(mindmap: Mindmap, block_id: str) -> ConversationGraph:
    """Return the graph containing ``block_id`` or raise 404."""
    for graph in mindmap.graphs.values():
        if block_id in graph.blocks:
            return graph
    raise HTTPException(status_code=404, detail=f"Block {block_id} not found")


def serialize_messages(graph: ConversationGraph, block_id: str) -> List[Dict[str, Any]]:
    """Messages of a block in the JSON shape the frontend expects."""
    return [
        {
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
        }
        for msg in graph.get_block_messages(block_id)
    ]


@app.exception_handler(LLMResponseError)
async def llm_response_error_handler(request: Request, exc: LLMResponseError):
    """The model returned nothing usable (e.g. a safety block): report it, don't 500."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})


# ============= Request/Response Models =============

class StartConversationRequest(BaseModel):
    """Request to start a new conversation."""
    topic: str


class MessageRequest(BaseModel):
    """Request to add a message to a block."""
    content: str


class MessageResponse(BaseModel):
    """Response containing block messages."""
    messages: List[Dict[str, Any]]
    current_block_id: str


class ChatRequest(BaseModel):
    """Generic chat request, mirroring CLI logic in main.py."""
    content: str


# ============= Frontend Pages =============

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render the home page."""
    mindmap = get_storage().load()
    mindmaps_list = []
    for graph_id, graph in mindmap.graphs.items():
        root_block = graph.blocks.get(graph.root_block_id)
        mindmaps_list.append({
            "graph_id": graph_id,
            "title": root_block.title if root_block else "Untitled",
        })

    # Request-first signature: the old (name, {"request": ...}) form was
    # removed in Starlette 1.0 and crashes the home page there.
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "mindmaps": mindmaps_list,
            "messages": [],
        },
    )


# ============= REST API Endpoints =============

@app.get("/api/mindmaps")
def list_mindmaps():
    """
    List all mindmaps (graphs).

    Returns:
        List of mindmap summaries with id, title, root_block_id
    """
    mindmap = get_storage().load()

    mindmaps_list = []
    for graph_id, graph in mindmap.graphs.items():
        root_block = graph.blocks.get(graph.root_block_id)
        mindmaps_list.append({
            "graph_id": graph_id,
            "title": root_block.title if root_block else "Untitled",
            "root_block_id": graph.root_block_id,
            "block_count": len(graph.blocks),
            "message_count": len(graph.messages),
            "is_current": graph_id == mindmap.current_graph_id,
        })

    return {"mindmaps": mindmaps_list}


@app.post("/api/mindmaps/new")
def create_new_mindmap(payload: StartConversationRequest):
    """
    Start a new conversation using ConversationManager.
    Calls manager.start_new_conversation(topic), titled with the topic.

    Args:
        payload: StartConversationRequest with topic

    Returns:
        New mindmap with root block and initial response
    """
    with write_lock:
        mgr = get_conversation_manager(get_storage().load())
        # The root node title is the user-provided topic (saved once, by the manager)
        response_text = mgr.start_new_conversation(payload.topic, title=payload.topic or None)

    graph = mgr.active_graph()
    root_block = graph.blocks[graph.root_block_id]
    return {
        "graph_id": graph.graph_id,
        "title": root_block.title,
        "root_block_id": root_block.block_id,
        "block_count": len(graph.blocks),
        "message_count": len(graph.messages),
        "is_current": True,
        "initial_response": response_text,
    }


@app.post("/api/chat")
def chat(payload: ChatRequest):
    """Simple chat endpoint using ConversationManager like the CLI.

    Logic is equivalent to:

        if not manager.graph or not manager.graph.root_block_id:
            response = manager.start_new_conversation(user_input)
        else:
            response = manager.continue_conversation(user_input)
    """
    with write_lock:
        mgr = get_conversation_manager(get_storage().load())

        # Decide whether to start a new conversation or continue the current one
        if not mgr.graph or not mgr.graph.root_block_id:
            assistant_response = mgr.start_new_conversation(payload.content)
        else:
            assistant_response = mgr.continue_conversation(payload.content)
        # ConversationManager has saved to storage.

    graph = mgr.active_graph()
    current_block_id = graph.current_block_id or graph.root_block_id
    return {
        "graph_id": graph.graph_id,
        "current_block_id": current_block_id,
        "graph": graph.to_d3_graph(),
        "messages": serialize_messages(graph, current_block_id) if current_block_id else [],
        "assistant_response": assistant_response,
    }


@app.get("/api/mindmaps/{graph_id}/graph")
def get_graph(graph_id: str):
    """
    Get full graph data for D3 visualization.

    Args:
        graph_id: ID of the graph

    Returns:
        D3-formatted graph with nodes and links
    """
    mindmap = get_storage().load()
    graph = mindmap.graphs.get(graph_id)

    if not graph:
        raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")

    return graph.to_d3_graph()


@app.get("/api/blocks/{block_id}/messages")
def get_block_messages(block_id: str):
    """
    Get all messages for a specific block.

    Args:
        block_id: ID of the block

    Returns:
        List of messages in the block
    """
    graph = find_block_graph(get_storage().load(), block_id)
    block = graph.blocks[block_id]

    return {
        "block_id": block_id,
        "title": block.title,
        "intent": block.intent,
        "summary": block.summary,
        "messages": serialize_messages(graph, block_id),
    }


@app.post("/api/blocks/{block_id}/messages")
def add_message_to_block(block_id: str, payload: MessageRequest):
    """
    Add a user message to a block and get LLM response using ConversationManager.
    Calls manager.continue_conversation(content).

    The message may be routed to a different block (e.g. a new child on DEEPEN);
    ``current_block_id`` and ``messages`` describe the block it landed in.

    Args:
        block_id: ID of the block
        payload: MessageRequest with message content

    Returns:
        Updated messages and assistant response
    """
    with write_lock:
        mindmap = get_storage().load()
        graph = find_block_graph(mindmap, block_id)
        mindmap.current_graph_id = graph.graph_id  # Switch to this graph
        graph.current_block_id = block_id

        mgr = get_conversation_manager(mindmap)
        try:
            response_text = mgr.continue_conversation(payload.content)  # saves
        except LLMResponseError:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}") from e

    graph = mgr.active_graph()  # may differ from the block's graph after a tangent
    current_block_id = graph.current_block_id
    return {
        "block_id": block_id,
        "graph_id": graph.graph_id,
        "current_block_id": current_block_id,
        "messages": serialize_messages(graph, current_block_id),
        "assistant_response": response_text,
    }


@app.post("/api/blocks/{block_id}/switch")
def switch_to_block(block_id: str):
    """
    Switch the active block (for UI context).

    Args:
        block_id: ID of the block to switch to

    Returns:
        Updated current block info
    """
    with write_lock:
        mindmap = get_storage().load()
        graph = find_block_graph(mindmap, block_id)
        mindmap.current_graph_id = graph.graph_id
        graph.current_block_id = block_id
        get_storage().save(mindmap)

    return {
        "block_id": block_id,
        "graph_id": graph.graph_id,
        "success": True,
    }


@app.delete("/api/blocks/{block_id}")
def delete_block(block_id: str):
    """
    Delete a block and all its descendants.
    """
    with write_lock:
        mindmap = get_storage().load()
        graph = find_block_graph(mindmap, block_id)
        mindmap.current_graph_id = graph.graph_id

        mgr = get_conversation_manager(mindmap)
        try:
            mgr.delete_block(block_id)  # saves
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    current_block_id = graph.current_block_id or graph.root_block_id
    return {
        "graph_id": graph.graph_id,
        "current_block_id": current_block_id,
        "graph": graph.to_d3_graph(),
        "messages": serialize_messages(graph, current_block_id) if current_block_id else [],
    }


@app.post("/api/mindmaps/{graph_id}/switch")
def switch_mindmap(graph_id: str):
    """
    Switch the active mindmap (graph).

    Args:
        graph_id: ID of the graph to switch to

    Returns:
        Updated current mindmap info
    """
    with write_lock:
        mindmap = get_storage().load()

        if graph_id not in mindmap.graphs:
            raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")

        mindmap.current_graph_id = graph_id
        get_storage().save(mindmap)

    return {
        "graph_id": graph_id,
        "success": True,
    }


@app.delete("/api/mindmaps/{graph_id}")
def delete_mindmap(graph_id: str):
    """
    Delete a mindmap (graph) and all its blocks/messages.
    """
    with write_lock:
        mindmap = get_storage().load()

        if graph_id not in mindmap.graphs:
            raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")

        del mindmap.graphs[graph_id]

        if mindmap.current_graph_id == graph_id:
            mindmap.current_graph_id = next(iter(mindmap.graphs.keys()), "")

        get_storage().save(mindmap)

    return {
        "success": True,
        "current_graph_id": mindmap.current_graph_id,
    }
