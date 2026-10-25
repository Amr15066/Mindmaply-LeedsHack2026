"""Test helpers: a scripted, offline LLM client and graph builders."""

import hashlib
import json
import math
import random
from typing import Any, Callable, Dict, List, Optional

from mindmap_chat.llm.base import LLMClient
from mindmap_chat.models import Block, ConversationGraph, ConversationMessage

# Markers that identify each prompt in mindmap_chat/llm/prompts.py.
CLASSIFY = "Classify this message as one of"
SUMMARY = "Summarize the discussion in this block"
INTENT = "The user is starting a new discussion thread"
ANSWER = "You are having a focused discussion"


def unit_vector(seed: str, dim: int = 64) -> List[float]:
    """Deterministic pseudo-random unit vector; distinct seeds are near-orthogonal."""
    rng = random.Random(hashlib.sha256(seed.encode()).hexdigest())
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def blend(a: List[float], b: List[float], cos: float) -> List[float]:
    """Return a unit vector with cosine similarity ~``cos`` to ``a`` (b must be ~orthogonal to a)."""
    # Gram-Schmidt so the result is exact even if b is not perfectly orthogonal.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    perp = [y - dot * x for x, y in zip(a, b, strict=True)]
    norm = math.sqrt(sum(p * p for p in perp))
    perp = [p / norm for p in perp]
    sin = math.sqrt(max(0.0, 1.0 - cos * cos))
    return [cos * x + sin * p for x, p in zip(a, perp, strict=True)]


class FakeLLMClient(LLMClient):
    """
    Offline LLMClient. Responses are chosen by which prompt template is being used.

    - ``classification`` / ``summary`` / ``intent``: dicts returned (as JSON) for those prompts
    - ``answer``: text (or callable(prompt) -> text) for answer prompts
    - ``embeddings``: explicit text -> vector map; anything else gets ``unit_vector(text)``
    """

    def __init__(self) -> None:
        self.classification: Dict[str, Any] = {
            "classification": "CONTINUE",
            "confidence": 0.5,
            "reasoning": "fake",
            "new_blocks": [],
        }
        self.summary: Dict[str, Any] = {
            "summary": "fake summary",
            "key_points": ["kp"],
            "open_questions": ["oq"],
            "title_suggestion": None,
        }
        self.intent: Dict[str, Any] = {"intent": "fake intent", "title": "Fake Title"}
        self.answer: str | Callable[[str], str] = "fake answer"
        self.raw_responses: Optional[List[str]] = None  # if set, call() pops from this first
        self.embeddings: Dict[str, List[float]] = {}
        self.prompts: List[str] = []
        self.embedded: List[str] = []
        self.batches: List[List[str]] = []

    def call(self, prompt: str, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        if self.raw_responses:
            return self.raw_responses.pop(0)
        if CLASSIFY in prompt:
            return json.dumps(self.classification)
        if SUMMARY in prompt:
            return json.dumps(self.summary)
        if INTENT in prompt:
            return json.dumps(self.intent)
        if callable(self.answer):
            return self.answer(prompt)
        return self.answer

    def embed(self, text: str) -> List[float]:
        self.embedded.append(text)
        if text in self.embeddings:
            return list(self.embeddings[text])
        return unit_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.batches.append(list(texts))
        return [self.embed(text) for text in texts]

    def prompts_of(self, marker: str) -> List[str]:
        return [p for p in self.prompts if marker in p]


def make_tree():
    """root -> (a -> (a1, a2), b)"""
    graph = ConversationGraph()
    root = Block(title="root", intent="root intent")
    graph.add_block(root)
    a = Block(parent_block_id=root.block_id, title="a")
    graph.add_block(a)
    a1 = Block(parent_block_id=a.block_id, title="a1")
    graph.add_block(a1)
    a2 = Block(parent_block_id=a.block_id, title="a2")
    graph.add_block(a2)
    b = Block(parent_block_id=root.block_id, title="b")
    graph.add_block(b)
    return graph, root, a, a1, a2, b


def add_message(graph, block, role="user", content="hi"):
    msg = ConversationMessage(block_id=block.block_id, role=role, content=content)
    graph.add_message(msg)
    block.add_message_ref(msg.message_id)
    return msg
