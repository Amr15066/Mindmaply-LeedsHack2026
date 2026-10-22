"""
Utility functions.
"""

from typing import List
from mindmap_chat.models import Block, ConversationGraph


def print_block_tree(graph: ConversationGraph, block_id: str = None, indent: int = 0) -> None:
    """
    Print the block tree to console.
    
    Args:
        graph: Conversation graph
        block_id: Block to start from (uses root if None)
        indent: Indentation level
    """
    if block_id is None:
        block_id = graph.root_block_id
    
    if not block_id or block_id not in graph.blocks:
        return
    
    block = graph.blocks[block_id]
    prefix = "  " * indent
    
    msg_count = len(block.conversation_refs)
    print(f"{prefix}[BLOCK] {block.title}")
    print(f"{prefix}   Intent: {block.intent}")
    print(f"{prefix}   Messages: {msg_count}")
    
    # Print children
    for child_id in block.children:
        print_block_tree(graph, child_id, indent + 1)


def get_all_blocks_in_order(graph: ConversationGraph) -> List[Block]:
    """
    Get all blocks in depth-first order.
    
    Args:
        graph: Conversation graph
        
    Returns:
        List of blocks
    """
    blocks = []
    
    def visit(block_id: str):
        if block_id not in graph.blocks:
            return
        block = graph.blocks[block_id]
        blocks.append(block)
        for child_id in block.children:
            visit(child_id)
    
    visit(graph.root_block_id)
    return blocks
