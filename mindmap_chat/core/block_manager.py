"""
Block lifecycle management.
Creating, updating, and summarizing blocks.
"""

from typing import Optional
from mindmap_chat.llm.base import LLMClient
from mindmap_chat.llm import prompts
from mindmap_chat.models import Block, ConversationGraph, ConversationMessage
from mindmap_chat.core.embeddings import embed_text
from mindmap_chat.core.context_builder import construct_summary_prompt_context
from mindmap_chat.config import config


def create_root_block(llm_client: LLMClient, user_message: str) -> Block:
    """
    Create the root block from first user message.
    Extracts intent and generates title.
    
    Args:
        llm_client: LLM client
        user_message: The first user message
        
    Returns:
        New Block instance
    """
    # Extract intent from message
    prompt = prompts.prompt_extract_intent_from_message(user_message)
    response = llm_client.call_json(prompt)
    
    intent = response.get("intent", "Initial conversation")
    title = response.get("title", "Untitled")
    
    # Embed the intent
    intent_embedding = embed_text(llm_client, intent)
    
    # Create block
    block = Block(
        title=title,
        intent=intent,
        embedding=intent_embedding,
    )
    
    return block


def create_child_block(llm_client: LLMClient, parent_block: Block, 
                      title: str, intent: str) -> Block:
    """
    Create a child block.
    
    Args:
        llm_client: LLM client for embedding
        parent_block: Parent block
        title: Block title
        intent: Block intent
        
    Returns:
        New Block instance
    """
    # Embed the intent
    intent_embedding = embed_text(llm_client, intent)
    
    # Create block
    block = Block(
        parent_block_id=parent_block.block_id,
        title=title,
        intent=intent,
        embedding=intent_embedding,
    )
    
    # Update parent
    parent_block.add_child(block.block_id)
    
    return block


def summarize_block(llm_client: LLMClient, graph: ConversationGraph, 
                   block: Block) -> None:
    """
    Auto-summarize a block using LLM.
    Updates block in-place with summary, key_points, open_questions.
    
    Args:
        llm_client: LLM client
        graph: Conversation graph
        block: Block to summarize
    """
    # Get context for summarization
    context = construct_summary_prompt_context(graph, block)
    
    # Call LLM
    prompt = prompts.prompt_generate_block_summary(block.intent, context)
    
    try:
        response = llm_client.call_json(prompt)
        
        # Update block
        block.summary = response.get("summary", "")
        block.key_points = response.get("key_points", [])
        block.open_questions = response.get("open_questions", [])
        
        # Update title if suggested
        new_title = response.get("title_suggestion")
        if new_title:
            block.title = new_title
        
        print(f"[OK] Block '{block.title}' summarized")
    
    except Exception as e:
        print(f"Error summarizing block: {e}")


def maybe_auto_summarize(llm_client: LLMClient, graph: ConversationGraph, 
                        block: Block) -> None:
    """
    Automatically summarize block if it has enough messages.
    
    Args:
        llm_client: LLM client
        graph: Conversation graph
        block: Block to check
    """
    message_count = len(block.conversation_refs)
    threshold = config.auto_summarize_after_n_messages
    
    if message_count >= threshold and not block.summary:
        print(f"Auto-summarizing block (reached {message_count} messages)...")
        summarize_block(llm_client, graph, block)
