"""
Intent shift detection.
Determines if new message continues, deepens, or diverges from current block.
"""

import json
from typing import List, Optional

from mindmap_chat.llm.base import LLMClient
from mindmap_chat.llm import prompts
from mindmap_chat.models import Block, BlockClassification, ConversationMessage
from mindmap_chat.core.embeddings import compute_similarity, embed_text
from mindmap_chat.config import config


def detect_intent_shift(llm_client: LLMClient, current_block: Block,
                       new_user_msg: str, last_messages: list[ConversationMessage],
                       new_msg_embedding: Optional[List[float]] = None) -> BlockClassification:
    """
    Detect if the new message represents an intent shift.

    Uses embedding similarity first (fast), then LLM classification if ambiguous.

    Args:
        llm_client: LLM client for embeddings and classification
        current_block: The current block
        new_user_msg: The new user message
        last_messages: Recent messages (for context)
        new_msg_embedding: Embedding of ``new_user_msg`` if the caller already
            has it (avoids embedding the same text twice in one turn)

    Returns:
        BlockClassification with action and reasoning
    """

    # Step 1: Embed the new message
    if new_msg_embedding is None:
        new_msg_embedding = embed_text(llm_client, new_user_msg)
    
    # Step 2: Compare similarity to current block intent
    intent_similarity = compute_similarity(new_msg_embedding, current_block.embedding)
    
    # Step 3: Make decision based on thresholds
    thresholds = config.thresholds
    
    if intent_similarity >= thresholds.continue_threshold:
        # Very high similarity: same topic
        return BlockClassification(
            action="continue",
            confidence=intent_similarity,
            reasoning=f"Message aligns strongly with block intent (similarity: {intent_similarity:.2f})"
        )
    
    elif intent_similarity >= thresholds.deepen_threshold:
        # Medium-high similarity: deeper dive
        return BlockClassification(
            action="deepen",
            confidence=intent_similarity,
            reasoning=f"Message deepens the current topic (similarity: {intent_similarity:.2f})"
        )
    
    else:
        # Below the deepen threshold: ambiguous or a topic shift, ask the LLM
        return _classify_with_llm(
            llm_client, current_block, new_user_msg, last_messages
        )


def _classify_with_llm(llm_client: LLMClient, current_block: Block, 
                       new_user_msg: str, last_messages: list[ConversationMessage]) -> BlockClassification:
    """
    Use LLM to classify intent shift when embedding similarity is ambiguous.
    """
    
    # Format last messages for context
    last_user = last_messages[-2].content if len(last_messages) >= 2 else "(first message)"
    last_assistant = last_messages[-1].content if last_messages else "(no response yet)"
    
    # Get classification from LLM
    base_prompt = prompts.prompt_classify_intent_shift(
        current_block.title,
        current_block.intent,
        current_block.summary or "(block just started)",
        last_user,
        last_assistant,
        new_user_msg
    )
    
    try:
        try:
            response_json = llm_client.call_json(base_prompt)
        except json.JSONDecodeError:
            retry_prompt = (
                base_prompt
                + "\n\nReminder: Return a single valid JSON object only. No extra text."
            )
            response_json = llm_client.call_json(retry_prompt)
        return _classification_from_json(response_json)

    except Exception as e:
        print(f"Error in LLM classification: {e}")
        # Fallback
        return BlockClassification(
            action="continue",
            confidence=0.5,
            reasoning="Fallback classification due to LLM error"
        )


_ACTION_MAP = {
    "CONTINUE": "continue",
    "DEEPEN": "deepen",
    "NEW_CHILD": "new_child",
    "TANGENT": "tangent",
}


def _classification_from_json(response_json: dict) -> BlockClassification:
    """Map the classifier's JSON onto a BlockClassification (unknown labels -> continue)."""
    llm_action = str(response_json.get("classification", "")).upper()
    action = _ACTION_MAP.get(llm_action, "continue")

    new_blocks = _parse_new_blocks(response_json)
    if not new_blocks:
        legacy_title = response_json.get("new_block_title")
        legacy_intent = response_json.get("new_block_intent")
        if legacy_title or legacy_intent:
            new_blocks = [{
                "title": legacy_title or "Untitled",
                "intent": legacy_intent or "New discussion",
            }]

    return BlockClassification(
        action=action,
        confidence=float(response_json.get("confidence", 0.5)),
        reasoning=response_json.get("reasoning", ""),
        new_block_title=response_json.get("new_block_title"),
        new_block_intent=response_json.get("new_block_intent"),
        new_blocks=new_blocks
    )


def _parse_new_blocks(response_json: dict) -> list[dict]:
    new_blocks = []
    for item in response_json.get("new_blocks", []) or []:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        intent = item.get("intent")
        if not title and not intent:
            continue
        new_blocks.append({
            "title": title or "Untitled",
            "intent": intent or "New discussion",
        })
    return new_blocks
