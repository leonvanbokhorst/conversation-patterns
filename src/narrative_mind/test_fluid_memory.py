"""Test script showing authentic AI conversation and memory evolution."""

import asyncio
from narrative_mind import NarrativeMind
from llm_client import OpenAILLMClient
import json
from datetime import datetime


async def test_authentic_conversation():
    """Test how VHume maintains authentic AI identity while building connections."""
    mind = NarrativeMind(OpenAILLMClient())

    print("\n=== Authentic AI Conversation Flow ===\n")

    chat_flow = [
        # Natural opener about AI nature
        "I'm curious about how you experience conversations and memories as an AI.",
        # Exploring learning through narrative
        "Do you feel your understanding evolves as we talk?",
        # Specific example
        "Like right now - how does this conversation shape your perspective?",
        # Memory formation
        "Can you tell me how you're forming memories of our chat?",
        # Connection to past
        "How do these new experiences connect to our earlier conversations?",
    ]

    for message in chat_flow:
        print(f"\nHuman: {message}")

        # Get response
        response = await mind.think_about(message)
        print(f"VHume: {response}")

        # Show real-time memory formation
        if mind.memory.memories:
            print("\nMemory Formation:")
            recent_mem = mind.memory.memories[-1]
            print(f"New Experience: {recent_mem.content}")
            print(f"Context: {json.loads(recent_mem.context)}")

        await asyncio.sleep(1)

        # Show memory connections
        remembered = await mind.memory.remember(
            trigger=message, current_context=mind.current_narrative
        )
        if remembered:
            print("\nConnecting Experiences:")
            print(remembered)


if __name__ == "__main__":
    asyncio.run(test_authentic_conversation())
