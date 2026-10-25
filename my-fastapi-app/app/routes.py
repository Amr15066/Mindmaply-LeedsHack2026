from fastapi import Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from app import app, templates
from mindmap_chat.storage import JSONStorage
from mindmap_chat.conversation import ConversationManager
from mindmap_chat.llm.base import LLMResponseError
from mindmap_chat.llm.gemini import GeminiClient
from mindmap_chat.config import config, validate_config

# Initialize backends (lazy - only validate when actually needed).
# Keep LLM and storage cached, but ALWAYS create a fresh ConversationManager
# so it reloads the latest mindmap state from disk on every request.
storage = None
llm_client = None

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

def get_conversation_manager() -> ConversationManager:
    """Return a fresh ConversationManager bound to current storage state."""
    return ConversationManager(get_llm_client(), get_storage())


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
async def home(request: Request):
    """Render the home page."""
    mindmap = get_storage().load()
    mindmaps_list = []
    for graph_id, graph in mindmap.graphs.items():
        root_block = graph.blocks.get(graph.root_block_id)
        mindmaps_list.append({
            "graph_id": graph_id,
            "title": root_block.title if root_block else "Untitled",
        })

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
async def list_mindmaps():
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
async def create_new_mindmap(payload: StartConversationRequest):
    """
    Start a new conversation using ConversationManager.
    Calls manager.start_new_conversation(topic).
    
    Args:
        payload: StartConversationRequest with topic
        
    Returns:
        New mindmap with root block and initial response
    """
    mgr = get_conversation_manager()
    
    # Use manager to start conversation (this creates the graph structure)
    response_text = mgr.start_new_conversation(payload.topic)
    
    # Get the current graph (just created)
    mindmap = get_storage().load()
    graph = mindmap.graphs.get(mindmap.current_graph_id)
    root_block = graph.blocks.get(graph.root_block_id)

    # Ensure the root node title matches the user-provided topic
    if root_block and payload.topic:
        root_block.title = payload.topic
        get_storage().save(mindmap)
    
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
async def chat(payload: ChatRequest):
    """Simple chat endpoint using ConversationManager like the CLI.

    Logic is equivalent to:

        if not manager.graph or not manager.graph.root_block_id:
            response = manager.start_new_conversation(user_input)
        else:
            response = manager.continue_conversation(user_input)
    """
    mgr = get_conversation_manager()

    # Decide whether to start a new conversation or continue the current one
    if not mgr.graph or not mgr.graph.root_block_id:
        assistant_response = mgr.start_new_conversation(payload.content)
    else:
        assistant_response = mgr.continue_conversation(payload.content)

    # After the call above, ConversationManager has already saved to storage.
    graph = mgr.graph
    if not graph:
        # Should not happen, but keep response predictable
        raise HTTPException(status_code=500, detail="No active graph after chat call")

    current_block_id = graph.current_block_id or graph.root_block_id
    messages = graph.get_block_messages(current_block_id) if current_block_id else []
    messages_list = [
        {
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
        }
        for msg in messages
    ]

    return {
        "graph_id": graph.graph_id,
        "current_block_id": current_block_id,
        "graph": graph.to_d3_graph(),
        "messages": messages_list,
        "assistant_response": assistant_response,
    }


@app.get("/api/mindmaps/{graph_id}/graph")
async def get_graph(graph_id: str):
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
async def get_block_messages(block_id: str):
    """
    Get all messages for a specific block.
    
    Args:
        block_id: ID of the block
        
    Returns:
        List of messages in the block
    """
    mindmap = get_storage().load()
    
    # Find the graph containing this block
    graph = None
    for g in mindmap.graphs.values():
        if block_id in g.blocks:
            graph = g
            break
    
    if not graph:
        raise HTTPException(status_code=404, detail=f"Block {block_id} not found")
    
    block = graph.blocks[block_id]
    messages = graph.get_block_messages(block_id)
    
    # Convert to JSON-serializable format
    messages_list = [
        {
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
        }
        for msg in messages
    ]
    
    return {
        "block_id": block_id,
        "title": block.title,
        "intent": block.intent,
        "summary": block.summary,
        "messages": messages_list,
    }


@app.post("/api/blocks/{block_id}/messages")
async def add_message_to_block(block_id: str, payload: MessageRequest, background_tasks: BackgroundTasks):
    """
    Add a user message to a block and get LLM response using ConversationManager.
    Calls manager.continue_conversation(content).
    
    Args:
        block_id: ID of the block
        payload: MessageRequest with message content
        
    Returns:
        Updated messages and assistant response
    """
    mindmap = get_storage().load()
    
    # Find the graph containing this block
    graph = None
    for gid, g in mindmap.graphs.items():
        if block_id in g.blocks:
            graph = g
            mindmap.current_graph_id = gid  # Switch to this graph
            break
    
    if not graph:
        raise HTTPException(status_code=404, detail=f"Block {block_id} not found")
    
    # Update conversation manager context and use it to continue conversation
    mgr = get_conversation_manager()
    mgr.mindmap = mindmap
    mgr.graph = graph
    mgr.graph.current_block_id = block_id
    
    # Use manager to continue conversation
    try:
        mgr.continue_conversation(payload.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}") from e
    
    # Save updated graph
    get_storage().save(mindmap)
    
    # Get updated messages for this block
    messages = graph.get_block_messages(block_id)
    messages_list = [
        {
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
        }
        for msg in messages
    ]
    
    return {
        "block_id": block_id,
        "messages": messages_list,
        "current_block_id": block_id,
    }


@app.post("/api/blocks/{block_id}/switch")
async def switch_to_block(block_id: str):
    """
    Switch the active block (for UI context).
    
    Args:
        block_id: ID of the block to switch to
        
    Returns:
        Updated current block info
    """
    mindmap = get_storage().load()
    
    # Find the graph containing this block
    for gid, g in mindmap.graphs.items():
        if block_id in g.blocks:
            mindmap.current_graph_id = gid
            g.current_block_id = block_id
            get_storage().save(mindmap)
            return {
                "block_id": block_id,
                "graph_id": gid,
                "success": True,
            }
    
    raise HTTPException(status_code=404, detail=f"Block {block_id} not found")


@app.delete("/api/blocks/{block_id}")
async def delete_block(block_id: str):
    """
    Delete a block and all its descendants.
    """
    mindmap = get_storage().load()

    graph = None
    for gid, g in mindmap.graphs.items():
        if block_id in g.blocks:
            graph = g
            mindmap.current_graph_id = gid
            break

    if not graph:
        raise HTTPException(status_code=404, detail=f"Block {block_id} not found")

    mgr = get_conversation_manager()
    mgr.mindmap = mindmap
    mgr.graph = graph

    try:
        mgr.delete_block(block_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Reload updated mindmap after deletion
    mindmap = get_storage().load()
    graph = mindmap.graphs.get(mindmap.current_graph_id)
    if not graph:
        raise HTTPException(status_code=500, detail="No active graph after delete")

    current_block_id = graph.current_block_id or graph.root_block_id
    messages = graph.get_block_messages(current_block_id) if current_block_id else []
    messages_list = [
        {
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
        }
        for msg in messages
    ]

    return {
        "graph_id": graph.graph_id,
        "current_block_id": current_block_id,
        "graph": graph.to_d3_graph(),
        "messages": messages_list,
    }


@app.post("/api/mindmaps/{graph_id}/switch")
async def switch_mindmap(graph_id: str):
    """
    Switch the active mindmap (graph).
    
    Args:
        graph_id: ID of the graph to switch to
        
    Returns:
        Updated current mindmap info
    """
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
async def delete_mindmap(graph_id: str):
    """
    Delete a mindmap (graph) and all its blocks/messages.
    """
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
