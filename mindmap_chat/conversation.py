"""
Main conversation loop and orchestration.
Ties together all modules for the core functionality.
"""

from typing import Optional
import re
from mindmap_chat.llm.base import LLMClient
from mindmap_chat.llm import prompts
from mindmap_chat.models import ConversationGraph, ConversationMessage, Block, BlockClassification
from mindmap_chat.core import (
    detect_intent_shift,
    create_root_block,
    create_child_block,
    maybe_auto_summarize,
    construct_block_context,
    compute_similarity,
    embed_text,
)
from mindmap_chat.config import config
from mindmap_chat.storage import JSONStorage
from mindmap_chat.utils import print_block_tree


class ConversationManager:
    """Manages a multi-block conversation."""

    def __init__(self, llm_client: LLMClient, storage: JSONStorage):
        """
        Initialize conversation manager.
        
        Args:
            llm_client: LLM client instance
            storage: Storage backend
        """
        self.llm = llm_client
        self.storage = storage
        self.mindmap = storage.load()
        self.graph = self.mindmap.get_current_graph()

    def start_new_conversation(self, user_message: str) -> str:
        """
        Start a completely new conversation.
        
        Args:
            user_message: First user message
            
        Returns:
            Assistant response
        """
        # Create new graph + root block
        self.graph = ConversationGraph()
        root_block = create_root_block(self.llm, user_message)
        self.graph.add_block(root_block)
        self.graph.current_block_id = root_block.block_id
        self.mindmap.add_graph(self.graph)

        # Get response, then store the exchange
        response = self._get_response_in_block(root_block, user_message)
        self._record_exchange(root_block, user_message, response)

        # Save
        self.storage.save(self.mindmap)
        
        print(f"\n[OK] Started new conversation: '{root_block.title}'")
        return response

    def continue_conversation(self, user_message: str) -> str:
        """
        Continue conversation in current block or create new block.
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant response
        """
        if not self.graph:
            return self.start_new_conversation(user_message)

        current_block = self.graph.blocks[self.graph.current_block_id]
        
        # Get recent messages for context
        block_messages = self.graph.get_block_messages(current_block.block_id)
        
        # Detect intent shift
        print("\n[Analyzing intent...]")
        classification = detect_intent_shift(
            self.llm,
            current_block,
            user_message,
            block_messages
        )
        
        print(f"  [ACTION] {classification.action} (confidence: {classification.confidence:.2f})")
        print(f"  [REASON] {classification.reasoning}")
        
        # Handle classification
        if classification.action == "continue":
            # Stay in current block
            target_block = current_block

        elif classification.action == "deepen":
            # Create child block(s) for a deeper dive
            new_blocks = self._resolve_deepen_blocks(classification, current_block, user_message)
            created_blocks = self._create_child_blocks(
                current_block,
                new_blocks,
                relation="deepen",
                confidence=classification.confidence,
                reasoning=classification.reasoning,
            )
            target_block = created_blocks[0]
            self.graph.current_block_id = target_block.block_id

        elif classification.action == "tangent":
            matched = self._find_matching_block_in_other_graphs(user_message)
            if matched:
                matched_graph, matched_block, similarity = matched
                if similarity >= config.thresholds.continue_threshold:
                    self.graph = matched_graph
                    self.mindmap.current_graph_id = matched_graph.graph_id
                    self.graph.current_block_id = matched_block.block_id
                    target_block = matched_block
                    print(
                        f"  [MATCH] Redirected tangent to '{matched_block.title}' "
                        f"(similarity: {similarity:.2f})"
                    )
                else:
                    self.graph = matched_graph
                    self.mindmap.current_graph_id = matched_graph.graph_id
                    self.graph.current_block_id = matched_block.block_id
                    new_blocks = self._resolve_deepen_blocks(
                        classification,
                        matched_block,
                        user_message,
                    )
                    created_blocks = self._create_child_blocks(
                        matched_block,
                        new_blocks,
                        relation="sibling",
                        confidence=similarity,
                        reasoning=(
                            f"Tangent matched existing block '{matched_block.title}' "
                            f"(similarity: {similarity:.2f})"
                        ),
                    )
                    target_block = created_blocks[0]
                    self.graph.current_block_id = target_block.block_id
                    print(
                        f"  [MATCH] Created child under '{matched_block.title}' "
                        f"(similarity: {similarity:.2f})"
                    )
            else:
                # Start a new graph with this message as the root block
                self.graph = ConversationGraph()
                root_block = create_root_block(self.llm, user_message)
                self.graph.add_block(root_block)
                self.graph.current_block_id = root_block.block_id
                self.mindmap.add_graph(self.graph)
                target_block = root_block

        elif classification.action == "new_child":
            # Create new block(s)
            new_blocks = classification.new_blocks or []
            if not new_blocks and (classification.new_block_title or classification.new_block_intent):
                new_blocks = [{
                    "title": classification.new_block_title or "Untitled",
                    "intent": classification.new_block_intent or "New discussion",
                }]
            if not new_blocks:
                new_blocks = [{"title": "Untitled", "intent": "New discussion"}]

            created_blocks = self._create_child_blocks(
                current_block,
                new_blocks,
                relation="child",
                confidence=classification.confidence,
                reasoning=classification.reasoning,
            )
            target_block = created_blocks[0]
            self.graph.current_block_id = target_block.block_id
        
        else:
            # Fallback
            target_block = current_block
        
        # Get response, then store the exchange
        response = self._get_response_in_block(target_block, user_message)
        self._record_exchange(target_block, user_message, response)

        # Auto-summarize if needed
        maybe_auto_summarize(self.llm, self.graph, target_block)
        
        # Save
        self.storage.save(self.mindmap)
        
        return response

    def _get_response_in_block(self, block: Block, user_message: str) -> str:
        """
        Get LLM response while maintaining block context.
        
        Args:
            block: Current block
            user_message: User's message
            
        Returns:
            Assistant response
        """
        # Block-scoped context (title, intent, summary, key points, open
        # questions, recent history). Built before the new message is stored,
        # so the message appears in the prompt exactly once.
        context = construct_block_context(self.graph, block)
        prompt = prompts.prompt_answer_in_block_context(context, block.intent, user_message)
        return self.llm.call(prompt)

    def _record_exchange(self, block: Block, user_message: str, response: str) -> None:
        """Store a user message and the assistant's response in a block."""
        for role, content in (("user", user_message), ("assistant", response)):
            message = ConversationMessage(block_id=block.block_id, role=role, content=content)
            self.graph.add_message(message)
            block.add_message_ref(message.message_id)

    def _resolve_deepen_blocks(
        self,
        classification: "BlockClassification",
        current_block: Block,
        user_message: str,
    ) -> list[dict[str, str]]:
        new_blocks = classification.new_blocks or []
        if new_blocks:
            return new_blocks

        if classification.new_block_title or classification.new_block_intent:
            return [{
                "title": classification.new_block_title or "Untitled",
                "intent": classification.new_block_intent or "New discussion",
            }]

        prompt = prompts.prompt_classify_intent_shift(
            current_block.title,
            current_block.intent,
            current_block.summary or "(block just started)",
            "(context omitted)",
            "(context omitted)",
            user_message,
        )
        try:
            response_json = self.llm.call_json(prompt)
            for item in response_json.get("new_blocks", []) or []:
                if not isinstance(item, dict):
                    continue
                title = item.get("title")
                intent = item.get("intent")
                if title or intent:
                    new_blocks.append({
                        "title": title or "Untitled",
                        "intent": intent or "New discussion",
                    })
        except Exception as exc:
            print(f"  [WARN] Could not expand deepen blocks: {exc}")

        if not new_blocks:
            new_blocks = [{
                "title": _make_deepen_title(current_block.title, user_message),
                "intent": f"Explore details of {current_block.intent}: {user_message}",
            }]

        return new_blocks

    def _create_child_blocks(
        self,
        parent_block: Block,
        new_blocks: list[dict[str, str]],
        relation: str,
        confidence: float,
        reasoning: str,
    ) -> list[Block]:
        """Create child blocks, recording why they are connected to the parent."""
        created_blocks = []
        for block_seed in new_blocks:
            new_block = create_child_block(
                self.llm,
                parent_block,
                block_seed.get("title", "Untitled"),
                block_seed.get("intent", "New discussion"),
            )
            new_block.relation = relation
            new_block.relation_confidence = confidence
            new_block.relation_reasoning = reasoning
            self.graph.add_block(new_block)
            created_blocks.append(new_block)
            print(f"  [NEW] Created new block: '{new_block.title}'")
        return created_blocks
    
   
    def _find_matching_block_in_other_graphs(
        self,
        user_message: str,
    ) -> Optional[tuple[ConversationGraph, Block, float]]:
        if not self.graph:
            return None

        user_embedding = embed_text(self.llm, user_message)
        best_match: Optional[tuple[ConversationGraph, Block, float]] = None

        for graph in self.mindmap.graphs.values():
            for block in graph.blocks.values():
                if not block.intent:
                    continue
                if not block.embedding:
                    block.embedding = embed_text(self.llm, block.intent)
                similarity = compute_similarity(user_embedding, block.embedding)
                if best_match is None or similarity > best_match[2]:
                    best_match = (graph, block, similarity)

        if best_match and best_match[2] >= config.thresholds.sibling_threshold:
            return best_match

        return None
    
    def switch_block(self, block_id: str) -> str:
        """
        Switch to a different block.
        
        Args:
            block_id: Block to switch to
            
        Returns:
            Summary of the block
        """
        if not self.graph or block_id not in self.graph.blocks:
            return "Block not found"
        
        self.graph.current_block_id = block_id
        block = self.graph.blocks[block_id]
        self.storage.save(self.mindmap)
        
        summary = f"\n[BLOCK] Switched to: {block.title}\n"
        summary += f"Intent: {block.intent}\n"
        summary += f"Summary: {block.summary or '(not yet summarized)'}\n"
        summary += f"Messages: {len(block.conversation_refs)}"
        
        return summary

    def print_mindmap(self) -> None:
        """Print the block tree."""
        print("\n[MINDMAP] CONVERSATION GRAPH:")
        if not self.graph:
            print("(no active graph)")
            return
        print_block_tree(self.graph)

    def export_graph(self) -> dict:
        """Export graph as dict (for visualization, etc)."""
        if not self.graph:
            return {}
        return self.graph.to_dict()

    def delete_block(self, block_id: str) -> None:
        """Delete a block from the graph."""
        if not self.graph:
            raise ValueError("No active graph")
        if block_id not in self.graph.blocks:
            raise ValueError(f"Block not found: {block_id}")
        
        if block_id == self.graph.root_block_id:
            raise ValueError("Cannot delete root block")
        
        block = self.graph.blocks[block_id]
        
        # Delete all children using depth-first traversal
        descendants = self._delete_children(block_id)
        delete_ids = [block_id] + descendants

        if block.parent_block_id and block.parent_block_id in self.graph.blocks:
            parent = self.graph.blocks[block.parent_block_id]
            parent.children = [child for child in parent.children if child not in delete_ids]

        self.graph.delete_blocks(delete_ids)

        if self.graph.current_block_id in delete_ids:
            self.graph.current_block_id = block.parent_block_id or self.graph.root_block_id

        self.storage.save(self.mindmap)
    
    def _delete_children(self, block_id: str) -> list[str]:
        """
        Iteratively collect all children of a block using depth-first traversal.
        Avoids recursion depth issues and memory leak risks.
        
        Args:
            block_id: ID of the block whose children to delete
            
        Returns:
            List of deleted descendant block IDs (excluding the block_id itself)
        """
        deleted_ids = []
        stack = [block_id]
        visited = set()
        
        # Build parent->children map in reverse (children->parents)
        while stack:
            current_id = stack.pop()
            
            if current_id in visited:
                continue  # Avoid infinite loops from circular references
            
            visited.add(current_id)
            block = self.graph.blocks.get(current_id)
            
            if block:
                stack.extend(block.children)
        
        # visited contains all reachable blocks from block_id
        # Remove the root block_id itself
        deleted_ids = list(visited - {block_id})
        
        return deleted_ids
        
    def list_graphs(self) -> list[tuple[str, str]]:
        summaries = []
        for graph_id, graph in self.mindmap.graphs.items():
            title = "Untitled graph"
            if graph.root_block_id and graph.root_block_id in graph.blocks:
                title = graph.blocks[graph.root_block_id].title
            summaries.append((graph_id, title))
        return summaries

    def switch_graph(self, graph_id: str) -> str:
        if graph_id not in self.mindmap.graphs:
            return "Graph not found"
        self.mindmap.current_graph_id = graph_id
        self.graph = self.mindmap.graphs[graph_id]
        self.storage.save(self.mindmap)
        title = "Untitled graph"
        if self.graph.root_block_id in self.graph.blocks:
            title = self.graph.blocks[self.graph.root_block_id].title
        return f"\n[GRAPH] Switched to: {title}\nGraph ID: {graph_id}"
    
    def delete_graph(self, graph_id: str) -> None:
        """
        Delete an entire graph and all its blocks.
        
        Args:
            graph_id: ID of the graph to delete
            
        Raises:
            ValueError: If graph not found or only one graph remains
        """
        if graph_id not in self.mindmap.graphs:
            raise ValueError(f"Graph not found: {graph_id}")
        
        if len(self.mindmap.graphs) == 1:
            raise ValueError("Cannot delete the only graph")
        
        # Remove graph from mindmap
        del self.mindmap.graphs[graph_id]
        
        # Switch to another graph if current was deleted
        if self.mindmap.current_graph_id == graph_id:
            # Pick the first available graph
            remaining_graph_ids = list(self.mindmap.graphs.keys())
            if remaining_graph_ids:
                self.mindmap.current_graph_id = remaining_graph_ids[0]
                self.graph = self.mindmap.graphs[self.mindmap.current_graph_id]
            else:
                # Fallback: create empty graph
                self.graph = ConversationGraph()
                self.mindmap.add_graph(self.graph)
        
        self.storage.save(self.mindmap)
    
def _make_deepen_title(parent_title: str, user_message: str) -> str:
    cleaned_title = re.sub(r"^(deep dive|deepen|details)\s*[:\-]\s*", "", parent_title, flags=re.I).strip()
    cleaned_title = cleaned_title or "Details"
    snippet = _summarize_message_fragment(user_message)
    if snippet:
        return f"{cleaned_title} — {snippet}"
    return f"{cleaned_title} (details)"


def _summarize_message_fragment(message: str, max_words: int = 8, max_chars: int = 60) -> str:
    if not message:
        return ""
    normalized = " ".join(message.strip().split())
    if not normalized:
        return ""
    words = normalized.split()
    truncated = " ".join(words[:max_words])
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars].rstrip()
    return truncated