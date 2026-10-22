"""
Context construction for LLM calls.
Builds focused context from block data instead of dumping entire history.
"""

from typing import List
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
    
    context = f"""
BLOCK CONTEXT:
Title: {block.title}
Intent: {block.intent}
Summary: {block.summary}

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
