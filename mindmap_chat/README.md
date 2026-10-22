# Mindmap Chat: Branching Conversations with LLMs

A conversational interface that replaces linear chat with a branching, block-based conversation system visualized as a mind map.

## The Problem

When users learn via chat with LLMs:
- Conversations become long and linear
- Users explore tangents and subtopics
- Context bloat makes answers unfocused
- Hard to return to original line of thinking

## The Solution

**Blocks** = focused discussion nodes  
**Graph** = non-linear conversation structure  
**Smart Context** = only relevant information per block  
**Auto-splitting** = AI detects intent shifts and creates new blocks

## Quick Start

### 1. Setup

From the repository root:

```bash
pip install -e .
```

### 2. Get Gemini API Key

1. Go to [ai.google.dev](https://ai.google.dev)
2. Click "Get API key"
3. Copy your API key

### 3. Create `.env` file

```bash
cp mindmap_chat/.env.example mindmap_chat/.env
# Edit mindmap_chat/.env and add your API key
```

### 4. Run

```bash
mindmap-chat            # or: python -m mindmap_chat.main
```

### 5. Try it

```
> /help                          # See commands
> What is quantum computing?     # Start conversation
> How do quantum gates work?     # Continue (same block or new?)
> /map                           # View mindmap
> /switch <block_id>            # Switch to another block
```

## Architecture

### Modules

- **`models.py`**: Data structures (Block, Message, Graph)
- **`llm/`**: LLM abstraction (base class + Gemini implementation)
- **`llm/prompts.py`**: All LLM prompts in one place
- **`core/`**: Business logic
  - `intent_detector.py`: Detect intent shifts
  - `block_manager.py`: Create/summarize blocks
  - `context_builder.py`: Construct minimal context
  - `embeddings.py`: Similarity matching
- **`storage/`**: JSON file storage
- **`conversation.py`**: Main orchestration loop
- **`main.py`**: CLI entry point

### Design Principles

✅ **Modular**: Swap Gemini for OpenAI/Anthropic by changing one file  
✅ **Stateless**: Everything stored in JSON (easy to inspect/debug)  
✅ **Explicit**: Prompts and thresholds are tunable  
✅ **Simple**: No complex async, multi-user, or cloud infrastructure  

## Conversation Flow

```
User message arrives
  ↓
[1] Embed message & compare to current block intent
  ↓
[2] If unclear, ask LLM: Continue/Deepen/New block/Tangent?
  ↓
[3] If new block → create child block
  ↓
[4] Construct block-scoped context (summary + intent + last N messages)
  ↓
[5] Call Gemini with context
  ↓
[6] Store message and response
  ↓
[7] After N messages, auto-summarize block
  ↓
[8] Repeat
```

## Configuration

Edit `config.py`:

- **Gemini model**: `model_name` (default: `gemini-1.5-flash`)
- **Thresholds**: `continue_threshold`, `deepen_threshold`, etc.
- **Auto-summarize**: After how many messages?
- **Context size**: How many recent messages to include?

## Data Storage

Conversations stored in `./data/conversation.json`:

```json
{
  "root_block_id": "uuid",
  "blocks": {
    "uuid": {
      "title": "How Transformers Work",
      "intent": "Understand transformer architecture",
      "summary": "...",
      "children": ["uuid2", "uuid3"]
    }
  },
  "messages": { ... },
  "current_block_id": "uuid"
}
```

## Extending

### Add a New LLM Provider

1. Create `llm/openai.py` extending `LLMClient`
2. Implement `call()` and `embed()` methods
3. Update `main.py` to instantiate your client
4. Done! Everything else works.

### Change Storage Backend

1. Create `storage/postgres.py` extending abstract interface
2. Implement `save()` and `load()`
3. Update `main.py`
4. Done!

### Adjust Prompts

All prompts in `llm/prompts.py`. Edit and re-run.

## Next Steps for Production

- [ ] Web UI (React + D3 for mindmap)
- [ ] Multi-user with auth
- [ ] Database backend (Postgres)
- [ ] RAG for external knowledge
- [ ] Block-level search
- [ ] Export to Obsidian/Notion

## Hackathon MVP Checklist

- [x] Core data structures
- [x] Gemini integration
- [x] Intent detection (embedding + LLM)
- [x] Block creation and management
- [x] Auto-summarization
- [x] Context-aware responses
- [x] JSON storage
- [x] CLI interface
- [ ] Web visualization (D3 mindmap)
- [ ] Demo narrative
