"""
Context construction for LLM calls.
Builds focused context from block data instead of dumping entire history.
"""

from typing import List, Optional
from mindmap_chat.models import Block, ConversationMessage, ConversationGraph
from mindmap_chat.config import config


def format_key_points(key_points: List[str]) -> str:
    """Format key points as bullet list."""
    if not key_points:
        return "(No key points yet)"
    return "\n".join(f"- {kp}" for kp in key_points)


def format_open_questions(open_questions: List[str]) -> str:
    """Format open questions as bullet list."""
    if not open_questions:
        return "(No open questions)"
    return "\n".join(f"- {oq}" for oq in open_questions)


def format_conversation_turns(messages: List[ConversationMessage]) -> str:
    """Format messages as conversation turns."""
    if not messages:
        return "(No messages yet)"
    
    turns = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        turns.append(f"{role}: {msg.content}")
    
    return "\n\n".join(turns)


def format_ancestor_context(graph: ConversationGraph, block: Block,
                            max_depth: Optional[int] = None,
                            max_chars: Optional[int] = None) -> str:
    """
    Describe the blocks this one branched from, nearest first, so a child block
    keeps the thread it came from.

    Each ancestor contributes its summary (or its intent, if not yet summarized).
    Stops after ``max_depth`` ancestors or once ``max_chars`` would be exceeded;
    the nearest parent is always included, truncated if necessary.

    Returns:
        A formatted section ending in a blank line, or "" for a root block
    """
    if max_depth is None:
        max_depth = config.ancestor_context_depth
    if max_chars is None:
        max_chars = config.ancestor_context_chars

    lines: List[str] = []
    used = 0
    seen = {block.block_id}
    parent_id = block.parent_block_id
    while parent_id and parent_id not in seen and len(lines) < max_depth:
        parent = graph.blocks.get(parent_id)
        if parent is None:
            break
        seen.add(parent_id)
        line = f"- {parent.title}: {parent.summary or parent.intent}"
        if used + len(line) > max_chars:
            if lines:
                break
            line = line[: max(max_chars - 3, 0)] + "..."
        lines.append(line)
        used += len(line)
        parent_id = parent.parent_block_id

    if not lines:
        return ""
    return (
        "\nPARENT TOPICS (this block branched from these, nearest first):\n"
        + "\n".join(lines)
        + "\n"
    )


def construct_block_context(graph: ConversationGraph, block: Block,
                          max_messages: int = None) -> str:
    """
    Construct minimal context for answering in a block.
    
    Args:
        graph: The conversation graph
        block: The current block
        max_messages: Max recent messages to include (uses config default if None)
        
    Returns:
        Formatted context string
    """
    if max_messages is None:
        max_messages = config.context_window_size
    
    # Get messages for this block
    block_messages = graph.get_block_messages(block.block_id)
    
    # Take last N messages
    recent_messages = block_messages[-max_messages:] if block_messages else []
    
    # Format components
    key_points_str = format_key_points(block.key_points)
    open_questions_str = format_open_questions(block.open_questions)
    recent_messages_str = format_conversation_turns(recent_messages)
    ancestors_str = format_ancestor_context(graph, block)

    context = f"""{ancestors_str}
BLOCK CONTEXT:
Title: {block.title}
Intent: {block.intent}
Summary so far: {block.summary or "(discussion just started)"}

KEY POINTS COVERED:
{key_points_str}

OPEN QUESTIONS FROM THIS DISCUSSION:
{open_questions_str}

CONVERSATION HISTORY (last {len(recent_messages)} messages):
{recent_messages_str}
"""
    
    return context


def construct_summary_prompt_context(graph: ConversationGraph, block: Block) -> str:
    """
    Construct context for summarizing a block.
    Includes all messages for that block.
    
    Args:
        graph: The conversation graph
        block: The block to summarize
        
    Returns:
        Formatted context string
    """
    block_messages = graph.get_block_messages(block.block_id)
    messages_str = format_conversation_turns(block_messages)
    
    return f"""BLOCK INTENT: {block.intent}

MESSAGES IN THIS BLOCK:
{messages_str}"""
