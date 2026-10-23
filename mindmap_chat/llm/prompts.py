"""
All LLM prompts in one place. Easy to iterate and test.
"""


def prompt_classify_intent_shift(block_title: str, block_intent: str, block_summary: str, 
                                 last_user_msg: str, last_assistant_msg: str, 
                                 new_user_msg: str) -> str:
    """Prompt A: Classify whether the new message continues, deepens, or diverges."""
    return f"""You are analyzing whether a user's new message represents a shift in topic.

CURRENT BLOCK INFO:
Title: {block_title}
Intent: {block_intent}
Summary so far: {block_summary}

LAST EXCHANGE IN THIS BLOCK:
User: {last_user_msg}
Assistant: {last_assistant_msg}

NEW USER MESSAGE:
{new_user_msg}

Classify this message as one of:
- CONTINUE: Same topic, no significant shift
- DEEPEN: Diving deeper into the same topic
- NEW_CHILD: A related but distinct subtopic
- TANGENT: Unrelated topic (should be separate discussion)

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "classification": "CONTINUE | DEEPEN | NEW_CHILD | TANGENT",
  "confidence": 0.0-1.0,
  "reasoning": "one sentence explanation",
  "new_blocks": [
    {{"title": "title for child/tangent block", "intent": "intent statement"}},
    {{"title": "title for another child", "intent": "intent statement"}}
  ]
}}

Constraints:
- Output must be a single JSON object and nothing else.
- Strings must be fully quoted and on one line (escape newlines).
- If there are no new blocks, return "new_blocks": [].
"""


def prompt_generate_block_summary(block_intent: str, conversation_turns: str) -> str:
    """Prompt B: Summarize a block after it's been discussed."""
    return f"""Summarize the discussion in this block.

BLOCK INTENT: {block_intent}

MESSAGES IN THIS BLOCK:
{conversation_turns}

Generate a JSON response with ONLY these fields (no markdown):
{{
  "summary": "2-3 sentence concise summary of what was discussed and what was concluded",
  "key_points": ["point 1", "point 2", "point 3"],
  "open_questions": ["unresolved question 1", "unresolved question 2"],
  "title_suggestion": "better title if needed (or null)"
}}

Constraints:
- Summary must be under 150 words
- Key points are actionable takeaways
- Open questions are next logical steps
"""


def prompt_extract_intent_from_message(user_msg: str) -> str:
    """Prompt C: Extract intent from first user message."""
    return f"""The user is starting a new discussion thread.

User's message:
{user_msg}

Extract the core intent and suggest a title.

Respond ONLY with valid JSON (no markdown):
{{
  "intent": "one-sentence statement of what the user wants to understand or achieve",
  "title": "short title (3-5 words)",
  "expected_subtopics": ["subtopic 1", "subtopic 2"]
}}"""


def prompt_answer_in_block_context(block_context: str, block_intent: str,
                                   new_user_msg: str) -> str:
    """
    Prompt D: Answer while maintaining block context and scope.

    ``block_context`` comes from ``core.context_builder.construct_block_context``.
    """
    return f"""You are having a focused discussion with the user within a specific topic.
{block_context}
USER'S NEW MESSAGE:
{new_user_msg}

Instructions:
1. Answer the user's question staying strictly within this block's scope
2. If they ask something outside this block's scope, acknowledge but redirect:
   "That sounds like a separate topic—we could explore that in a new thread. For now, let's focus on {block_intent}."
3. Keep your answer focused and concise (under 300 words unless asked for depth)
4. If appropriate, end with 1-2 clarifying questions to deepen the discussion

Answer naturally (not in JSON):"""
