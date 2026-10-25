"""
Main entry point: Interactive CLI for the mindmap chat system.
"""

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import sys
from mindmap_chat.config import config, validate_config
from mindmap_chat.llm.gemini import GeminiClient
from mindmap_chat.storage import JSONStorage
from mindmap_chat.conversation import ConversationManager


def print_help():
    """Print command help."""
    print("""
Commands:
  /new          Start a new conversation
  /map          View the mindmap
  /blocks       List all blocks
  /view <id>    Show messages for a block
  /switch <id>  Switch to a block
  /delete <id>  Delete a block
  /graphs       List all graphs
  /switch-graph <id>  Switch to a graph
  /delete-graph <id>  Delete an entire graph
  /clear        Clear conversation history
  /help         Show this help
  /exit         Exit
  
Just type to continue the current conversation.
""")


def main():
    """Main CLI loop."""
    # Routing decisions are logged at INFO, which the CLI shows by default.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="  [%(levelname)s] %(message)s",
    )
    try:
        validate_config()
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Initialize
    print("[INIT] Initializing Gemini Mindmap Chat...")
    llm = GeminiClient()
    storage = JSONStorage(config.storage_path)
    manager = ConversationManager(llm, storage)
    
    print("[OK] Ready!")
    print_help()
    
    # Main loop
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.split()[0]
                
                if cmd == "/help":
                    print_help()
                
                elif cmd == "/new":
                    print("Starting new conversation...")
                    first_msg = input("What do you want to discuss? > ").strip()
                    if first_msg:
                        response = manager.start_new_conversation(first_msg)
                        print(f"\nAssistant: {response}")
                
                elif cmd == "/map":
                    manager.print_mindmap()
                
                elif cmd == "/blocks":
                    graph_data = manager.export_graph()
                    if not graph_data:
                        print("\nBlocks:")
                        print("  (no active graph)")
                        continue
                    print("\nBlocks:")
                    for block_id, block_data in graph_data["blocks"].items():
                        print(f"  {block_id}: {block_data['title']}")
                
                elif cmd == "/switch":
                    try:
                        block_id = user_input.split()[1]
                        summary = manager.switch_block(block_id)
                        print(summary)
                    except IndexError:
                        print("Usage: /switch <block_id>")

                elif cmd == "/graphs":
                    graphs = manager.list_graphs()
                    print("\nGraphs:")
                    if not graphs:
                        print("  (none)")
                    for graph_id, title in graphs:
                        print(f"  {graph_id}: {title}")

                elif cmd == "/switch-graph":
                    try:
                        graph_id = user_input.split()[1]
                        summary = manager.switch_graph(graph_id)
                        print(summary)
                    except IndexError:
                        print("Usage: /switch-graph <graph_id>")
                
                elif cmd == "/delete":
                    try:
                        block_id = user_input.split()[1]
                        manager.delete_block(block_id)
                        print(f"[OK] Deleted block: {block_id}")
                    except IndexError:
                        print("Usage: /delete <block_id>")
                    except ValueError as e:
                        print(f"❌ Error: {e}")
                
                elif cmd == "/clear":
                    storage.clear()
                    manager.mindmap = storage.load()
                    manager.graph = manager.mindmap.get_current_graph()
                    print("[OK] Cleared")

                elif cmd == "/view":
                    try:
                        block_id = user_input.split()[1]
                        if not manager.graph or block_id not in manager.graph.blocks:
                            print("Block not found")
                            continue
                        messages = manager.graph.get_block_messages(block_id)
                        if not messages:
                            print("(no messages in this block)")
                            continue
                        print(f"\n[MESSAGES] Block: {manager.graph.blocks[block_id].title}\n")
                        for m in messages:
                            ts = m.timestamp
                            print(f"- [{m.role}] {m.content} ({ts})")
                    except IndexError:
                        print("Usage: /view <block_id>")
                
                elif cmd == "/exit":
                    print("Goodbye!")
                    break
                
                elif cmd == "/delete-graph":
                    try:
                        graph_id = user_input.split()[1]
                        print(f"[DEBUG] Attempting to delete graph: {graph_id}")
                        
                        # Ask for confirmation
                        graphs = manager.list_graphs()
                        print(f"[DEBUG] Available graphs: {graphs}")
                        
                        if not graphs:
                            print("❌ Error: No graphs available")
                            continue
                        
                        graph_title = next((title for gid, title in graphs if gid == graph_id), None)
                        if graph_title is None:
                            print(f"❌ Error: Graph ID '{graph_id}' not found")
                            print("Available graphs:")
                            for gid, title in graphs:
                                print(f"  {gid}: {title}")
                            continue
                        
                        confirm = input(f"\nDelete graph '{graph_title}'? This cannot be undone. (y/n): ").strip().lower()
                        if confirm == 'y':
                            print("[DEBUG] User confirmed deletion")
                            manager.delete_graph(graph_id)
                            print(f"[OK] Deleted graph: {graph_id}")
                        else:
                            print("[CANCELLED]")
                    except IndexError:
                        print("Usage: /delete-graph <graph_id>")
                        print("Use /graphs to see available graph IDs")
                    except ValueError as ve:
                        print(f"❌ Error: {ve}")
                    except Exception as e:
                        print(f"❌ Unexpected error: {e}")
                        import traceback
                        traceback.print_exc()
            
            else:
                # Regular conversation
                if not manager.graph or not manager.graph.root_block_id:
                    # No conversation started yet
                    response = manager.start_new_conversation(user_input)
                else:
                    # Continue conversation
                    response = manager.continue_conversation(user_input)
                
                print(f"\nAssistant: {response}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
