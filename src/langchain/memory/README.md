# CoALA Memory System

This implementation follows the Cognitive Architectures for Language Agents (CoALA) framework, providing a structured memory system for virtual humans and conversational agents.

## Features

- **Working Memory**: Maintains current context, goals, and attention focus
- **Episodic Memory**: Stores and retrieves past interactions with Redis
- **Tag-based Retrieval**: Search episodes by tags
- **Automatic Summarization**: Generates summaries of interactions
- **Async Support**: Asynchronous interaction processing

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start Redis server:

```bash
# Using Docker
docker run -d -p 6379:6379 redis

# Or use your system's Redis service
sudo service redis start
```

## Running Examples

The `examples.py` file demonstrates common usage patterns:

```bash
python examples.py
```

This will show:

1. Initializing the memory system
2. Processing conversations
3. Retrieving context for LLM prompts
4. Searching episodic memory
5. Managing working memory state

## Usage in Your Project

```python
from memory import CoALAMemorySystem, MemoryConfig

# Initialize
config = MemoryConfig(
    redis_url="redis://localhost:6379",
    summary_model="gpt-3.5-turbo"  # Optional
)
memory = CoALAMemorySystem(config)

# Process interactions
episode_id = await memory.process_interaction(
    content={
        "user": "Hello!",
        "assistant": "Hi there!"
    },
    tags=["greeting"]
)

# Get context for LLM
context = memory.get_context_for_llm()
```

## Architecture

The system follows the CoALA paper's memory architecture:

- `working_memory.py`: Short-term memory management
- `episodic_memory.py`: Long-term experience storage
- `memory_system.py`: Central coordinator

Future additions will include:

- Semantic memory for knowledge storage
- Procedural memory for skills/actions
