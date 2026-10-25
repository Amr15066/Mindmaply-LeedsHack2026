"""
Abstract base class for LLM clients.
Allows easy swapping of different LLM providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)

# Appended to free-text answers that hit the output-token limit, so a cut-off
# answer is never presented as complete.
TRUNCATION_NOTICE = "\n\n[Response truncated: reached the output-token limit]"


class LLMResponseError(RuntimeError):
    """The provider returned no usable content (e.g. safety block, empty candidates)."""


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def call(self, prompt: str, json_mode: bool = False) -> str:
        """
        Call the LLM with a prompt.
        
        Args:
            prompt: The prompt to send
            json_mode: If True, expect JSON-formatted response
            
        Returns:
            The LLM's response as a string
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed several texts. Override with a single batched request where the
        provider supports one; this default makes one call per text.
        """
        return [self.embed(text) for text in texts]

    def call_json(self, prompt: str) -> Dict[str, Any]:
        """
        Call the LLM and parse response as JSON.
        
        Args:
            prompt: The prompt
            
        Returns:
            Parsed JSON as dict
        """
        response = self.call(prompt, json_mode=True)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            extracted = _extract_json_payload(response)
            if extracted is not None:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    repaired = _repair_json_payload(extracted)
                    if repaired is not None:
                        return json.loads(repaired)
            repaired = _repair_json_payload(response)
            if repaired is not None:
                return json.loads(repaired)
            logger.error("Failed to parse JSON response: %r", response)
            raise

def _extract_json_payload(response: str) -> Optional[str]:
    if not response:
        return None
    stripped = response.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = response.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(response)):
        char = response[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return response[start:idx + 1].strip()
    return None


def _repair_json_payload(response: str) -> Optional[str]:
    if not response:
        return None
    start = response.find("{")
    if start == -1:
        return None
    snippet = response[start:]
    cleaned_chars = []
    depth = 0
    in_string = False
    escape = False
    for char in snippet:
        if in_string:
            if escape:
                escape = False
                cleaned_chars.append(char)
                continue
            if char == "\\":
                escape = True
                cleaned_chars.append(char)
                continue
            if char == "\n":
                cleaned_chars.append("\\n")
                continue
            if char == '"':
                in_string = False
                cleaned_chars.append(char)
                continue
            cleaned_chars.append(char)
            continue
        if char == '"':
            in_string = True
            cleaned_chars.append(char)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(depth - 1, 0)
        cleaned_chars.append(char)

    if in_string:
        cleaned_chars.append('"')
    if depth > 0:
        cleaned_chars.append("}" * depth)
    repaired = "".join(cleaned_chars).strip()
    if not repaired.startswith("{") or not repaired.endswith("}"):
        return None
    return repaired