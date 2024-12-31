"""Test script for memory chain implementation."""

import asyncio
from narrative_mind import NarrativeMind
from llm_client import OpenAILLMClient
import json
from pathlib import Path


async def test_memory_chains():
    """Test memory chains with a sequence of interactions."""
    mind = NarrativeMind(OpenAILLMClient())

    # Test sequence
    interactions = [
        "Hi! I'm Lonn. Nice to meet you!",
        "I love exploring how minds work, both human and artificial.",
        "What fascinates you most about consciousness?",
        "Do you remember what I said about minds earlier?",
        "You seem to have a good memory! How do you organize your thoughts?",
    ]

    print("\n=== Testing Memory Chains ===\n")

    for interaction in interactions:
        print(f"\nHuman: {interaction}")
        response = await mind.think_about(interaction)
        print(f"VHume: {response}")

        # Show memory analysis
        print("\nMemory Analysis:")
        memory_context = await mind.memory_store.get_memory_context(interaction)

        print("\nThemed Memories:")
        for memory in memory_context["themed_memories"]:
            print(f"- Theme: {memory['themes']}")
            print(f"  Content: {memory['content']}")
            print(f"  Emotions: {memory['emotional_context']}")

        print("\nRecent Chain:")
        for memory in memory_context["recent_chain"]:
            print(f"- Type: {memory['memory_type']}")
            print(f"  Content: {memory['content']}")

        await asyncio.sleep(1)  # Prevent rate limiting

    # Show final memory state
    chain_file = Path("memories/memory_chain.json")
    if chain_file.exists():
        print("\nFinal Memory Chain State:")
        print(json.dumps(json.loads(chain_file.read_text()), indent=2))


if __name__ == "__main__":
    asyncio.run(test_memory_chains())
