"""
Core data structures for the mindmap chat system.
These are JSON-serializable and represent the conversation graph.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime


@dataclass
class ConversationMessage:
    """A single message in the conversation."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    block_id: str = ""
    role: str = "user"  # "user" or "assistant"
    content: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    embedding: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMessage":
        return cls(**data)


@dataclass
class Block:
    """A node in the conversation graph (mindmap block)."""
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_block_id: Optional[str] = None
    title: str = ""
    intent: str = ""
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    embedding: List[float] = field(default_factory=list)  # Intent embedding
    children: List[str] = field(default_factory=list)
    conversation_refs: List[str] = field(default_factory=list)  # message_ids

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        return cls(**data)

    def add_message_ref(self, message_id: str):
        """Add a message ID reference to this block."""
        if message_id not in self.conversation_refs:
            self.conversation_refs.append(message_id)

    def add_child(self, block_id: str):
        """Add a child block."""
        if block_id not in self.children:
            self.children.append(block_id)


@dataclass
class ConversationGraph:
    """The entire conversation state."""
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    root_block_id: str = ""
    blocks: Dict[str, Block] = field(default_factory=dict)
    messages: Dict[str, ConversationMessage] = field(default_factory=dict)
    current_block_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "root_block_id": self.root_block_id,
            "blocks": {bid: block.to_dict() for bid, block in self.blocks.items()},
            "messages": {mid: msg.to_dict() for mid, msg in self.messages.items()},
            "current_block_id": self.current_block_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationGraph":
        blocks = {
            bid: Block.from_dict(block_data)
            for bid, block_data in data.get("blocks", {}).items()
        }
        messages = {
            mid: ConversationMessage.from_dict(msg_data)
            for mid, msg_data in data.get("messages", {}).items()
        }
        return cls(
            graph_id=data.get("graph_id", str(uuid.uuid4())),
            root_block_id=data.get("root_block_id", ""),
            blocks=blocks,
            messages=messages,
            current_block_id=data.get("current_block_id", ""),
            metadata=data.get("metadata", {}),
        )

    def add_block(self, block: Block):
        """Add a block to the graph."""
        if block.parent_block_id:
            if block.parent_block_id not in self.blocks:
                raise ValueError(
                    f"Parent block '{block.parent_block_id}' not found for '{block.block_id}'."
                )
            self.blocks[block.parent_block_id].add_child(block.block_id)
        else:
            if self.root_block_id and block.block_id != self.root_block_id:
                raise ValueError(
                    f"Root block already set to '{self.root_block_id}', "
                    f"cannot add another root '{block.block_id}'."
                )
            if not self.root_block_id:
                self.root_block_id = block.block_id
        self.blocks[block.block_id] = block

    def add_message(self, message: ConversationMessage):
        """Add a message to the graph."""
        self.messages[message.message_id] = message

    def get_block_messages(self, block_id: str) -> List[ConversationMessage]:
        """Get all messages for a block."""
        block = self.blocks.get(block_id)
        if not block:
            return []
        return [self.messages[mid] for mid in block.conversation_refs if mid in self.messages]
    
    def collect_descendants(self, block_id: str) -> List[str]:
        """Collect all descendant block IDs for a given block."""
        collected: List[str] = []
        to_visit = list(self.blocks.get(block_id, Block()).children)
        while to_visit:
            current_id = to_visit.pop()
            if current_id in collected:
                continue
            collected.append(current_id)
            current_block = self.blocks.get(current_id)
            if current_block:
                to_visit.extend(current_block.children)
        return collected

    def delete_blocks(self, block_ids: List[str]) -> None:
        """Delete blocks and their messages from the graph."""
        for block_id in block_ids:
            block = self.blocks.get(block_id)
            if not block:
                continue
            for message_id in block.conversation_refs:
                self.messages.pop(message_id, None)
            del self.blocks[block_id]

    def rebuild_children(self) -> None:
        """Rebuild children lists from parent_block_id references."""
        for block in self.blocks.values():
            block.children = []
        for block_id, block in self.blocks.items():
            if block.parent_block_id and block.parent_block_id in self.blocks:
                self.blocks[block.parent_block_id].add_child(block_id)

    def to_d3_graph(self) -> Dict[str, Any]:
        """
        Convert graph to D3.js-friendly format with relation metadata and colors.
        
        Returns:
            Dict with 'nodes' and 'links' for D3 visualization
        """
        # Relation type to color mapping
        relation_colors = {
            "continue": "#4CAF50",      # Green (same topic)
            "deepen": "#66BB6A",         # Light green (deeper in same topic)
            "child": "#2196F3",          # Blue (new subtopic)
            "sibling": "#FF9800",        # Orange (related sibling)
            "tangent": "#F44336",        # Red (unrelated tangent)
        }
        
        # Relation type to stroke width
        relation_weights = {
            "continue": 3,
            "deepen": 2.5,
            "child": 2,
            "sibling": 1.5,
            "tangent": 1,
        }
        
        nodes = []
        links = []
        
        # Build nodes from blocks
        for block_id, block in self.blocks.items():
            nodes.append({
                "id": block_id,
                "label": block.title or f"Block {block_id[:8]}",
                "intent": block.intent,
                "summary": block.summary,
                "key_points": block.key_points,
                "open_questions": block.open_questions,
                "message_count": len(block.conversation_refs),
                "is_current": block_id == self.current_block_id,
            })
        
        # Build links from parent-child relationships
        # Infer relation type from metadata (default to "child")
        for block_id, block in self.blocks.items():
            if block.parent_block_id:
                # Default to "child" relation with 0.8 confidence
                relation = "child"
                confidence = 0.8
                
                color = relation_colors.get(relation, relation_colors["child"])
                weight = relation_weights.get(relation, relation_weights["child"])
                
                links.append({
                    "source": block.parent_block_id,
                    "target": block_id,
                    "relation": relation,
                    "confidence": confidence,
                    "color": color,
                    "strokeWidth": weight,
                })
        
        return {
            "graph_id": self.graph_id,
            "root_block_id": self.root_block_id,
            "nodes": nodes,
            "links": links,
            "current_block_id": self.current_block_id,
        }

@dataclass
class BlockClassification:
    """Output from intent classifier."""
    action: str  # "continue" | "deepen" | "new_child" | "tangent"
    confidence: float
    reasoning: str
    new_block_title: Optional[str] = None
    new_block_intent: Optional[str] = None
    new_blocks: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
@dataclass
class Mindmap:
    """Container for multiple conversation graphs."""
    mindmap_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    graphs: Dict[str, ConversationGraph] = field(default_factory=dict)
    current_graph_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mindmap_id": self.mindmap_id,
            "graphs": {gid: graph.to_dict() for gid, graph in self.graphs.items()},
            "current_graph_id": self.current_graph_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Mindmap":
        graphs = {
            gid: ConversationGraph.from_dict(graph_data)
            for gid, graph_data in data.get("graphs", {}).items()
        }
        current_graph_id = data.get("current_graph_id", "")
        if current_graph_id and current_graph_id not in graphs:
            current_graph_id = ""
        return cls(
            mindmap_id=data.get("mindmap_id", str(uuid.uuid4())),
            graphs=graphs,
            current_graph_id=current_graph_id,
            metadata=data.get("metadata", {}),
        )

    def add_graph(self, graph: ConversationGraph) -> None:
        self.graphs[graph.graph_id] = graph
        self.current_graph_id = graph.graph_id

    def get_current_graph(self) -> Optional[ConversationGraph]:
        if not self.current_graph_id:
            return None
        return self.graphs.get(self.current_graph_id)
