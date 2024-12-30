"""Interactive terminal conversation with NarrativeMind."""

import asyncio
import sys
import time
from typing import Optional
from narrative_mind import NarrativeMind
from llm_client import OpenAILLMClient


def format_time(start_time: float) -> str:
    """Format elapsed time in milliseconds."""
    return f"{(time.time() - start_time) * 1000:.0f}ms"


async def initialize_mind() -> NarrativeMind:
    """Initialize NarrativeMind with OpenAI."""
    try:
        print("\nInitializing NarrativeMind...")
        start_time = time.time()
        mind = NarrativeMind(OpenAILLMClient())
        print(f"Initialization completed in {format_time(start_time)}")
        return mind
    except Exception as e:
        print(f"\nFailed to initialize OpenAI: {e}")
        sys.exit(1)


async def interactive_conversation(mind: NarrativeMind):
    """Run an interactive conversation with NarrativeMind."""
    print("\n=== Starting New Conversation ===")
    print("\nInitial mental model:", mind.mental_model)
    print("\nType 'quit' to end the conversation.")
    print("\n" + "=" * 50)

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit", "bye"]:
                break

            if not user_input:
                continue

            start_time = time.time()
            response = await mind.think_about(user_input)
            print(f"\nVHume: {response}")
            print(f"Time: {format_time(start_time)}")

    except KeyboardInterrupt:
        print("\n\nConversation ended by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


async def main():
    """Main entry point for the interactive conversation."""
    mind = await initialize_mind()
    await interactive_conversation(mind)


if __name__ == "__main__":
    asyncio.run(main())
