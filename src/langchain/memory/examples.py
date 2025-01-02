"""Examples demonstrating the CoALA memory system through a natural conversation."""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

from memory_system import CoALAMemorySystem
from procedural_memory import ActionStep
from config import MemoryConfig


# Example action functions
def greet_user(name: str, formal: bool = False) -> str:
    """Greet a user by name."""
    return f"Good day, {name}." if formal else f"Hi {name}! Nice to meet you!"


def note_interest(interest: str, context: Dict[str, Any]) -> None:
    """Note a user's interest in the context."""
    interests = context.setdefault("user_interests", [])
    if interest not in interests:
        interests.append(interest)


def suggest_topic(interests: List[str]) -> str:
    """Suggest a conversation topic based on interests."""
    if not interests:
        return "Could you tell me about your interests?"
    return f"I'd love to hear more about your interest in {interests[0]}!"


async def simulate_conversation(config: MemoryConfig = None):
    # sourcery skip: low-code-quality
    """Demonstrate the CoALA memory system through a natural conversation."""
    print("\n=== CoALA Memory System Demonstration ===")
    print(
        "This example shows how different types of memory work together in a conversation:"
    )
    print("- Working Memory: Keeps track of current context")
    print("- Episodic Memory: Remembers specific interactions")
    print("- Semantic Memory: Stores general knowledge")
    print("- Procedural Memory: Knows how to perform actions\n")

    # Initialize the memory system
    if not config:
        config = MemoryConfig(
            redis_url="redis://localhost:6379",
            working_memory_max_tokens=1000,
            summary_model="gpt-4o-mini",
            episodic_namespace="demo_agent",
            semantic_persist_dir="./semantic_memory",
            procedural_persist_dir="./procedural_memory",
            embedding_model="text-embedding-3-small",
        )

    memory = CoALAMemorySystem(config)

    # Register available actions
    memory.register_action("greet_user", greet_user)
    memory.register_action("note_interest", note_interest)
    memory.register_action("suggest_topic", suggest_topic)

    print("=== 1. Learning a New Procedure ===")
    print("First, the system learns how to greet users and engage with their interests")

    initial_procedure = {
        "timestamp": datetime.now().isoformat(),
        "action_sequence": {
            "name": "greet_and_engage",
            "description": "Welcome a user and engage with their interests",
            "steps": [
                {
                    "description": "Greet the user warmly",
                    "function": "greet_user",
                    "parameters": {"name": "{user_name}", "formal": False},
                },
                {
                    "description": "Remember their interest",
                    "function": "note_interest",
                    "parameters": {"interest": "{mentioned_interest}"},
                },
                {
                    "description": "Show interest in their hobby",
                    "function": "suggest_topic",
                    "parameters": {"interests": "{user_interests}"},
                },
            ],
            "context": {
                "usage": "For welcoming new users and building rapport",
            },
            "tags": ["greeting", "engagement"],
        },
    }

    await memory.process_interaction(
        content=initial_procedure, tags=["learning", "procedure_definition"]
    )
    print("✓ Learned a new conversation procedure")

    # Show initial working memory state
    print("\nInitial Working Memory:")
    print(memory.working_memory.load_memory_variables({}))

    print("\n=== 2. First User Interaction ===")
    print("Now someone new joins the conversation...")

    # First user interaction
    interaction1 = {
        "timestamp": datetime.now().isoformat(),
        "user": "Hi! I'm Alice and I love painting!",
        "assistant": "Hi Alice! Nice to meet you! I'd love to hear about your artistic journey!",
        "knowledge": "Alice is an artist who enjoys painting.",
    }

    await memory.process_interaction(content=interaction1, tags=["greeting", "art"])
    print("✓ Working Memory: Updated with current conversation context")
    print("✓ Episodic Memory: Stored the interaction with Alice")
    print("✓ Semantic Memory: Learned that Alice is an artist")

    # Show working memory after first interaction
    print("\nWorking Memory After Greeting:")
    working_state = memory.working_memory.load_memory_variables({})
    print(f"Current Context: {working_state}")

    print("\n=== 3. Using Learned Procedures ===")
    print("The system recognizes it can use its greeting procedure...")

    if procedures := memory.search_procedures(query="how to greet a new user", limit=1):
        proc = procedures[0]
        print(f"Found procedure: {proc['name']}")

        # Execute with Alice's context
        context = {
            "user_name": "Alice",
            "mentioned_interest": "painting",
            "user_interests": ["painting"],
        }

        success = await memory.execute_procedure(proc["id"], context)
        print("✓ Procedural Memory: Successfully executed greeting procedure")

        # Show working memory after procedure execution
        print("\nWorking Memory After Procedure:")
        print(memory.working_memory.load_memory_variables({}))

    print("\n=== 4. Continued Conversation ===")
    print("Alice shares more about her interests...")

    # Second interaction with more detail
    interaction2 = {
        "timestamp": datetime.now().isoformat(),
        "user": "I've been focusing on landscape painting lately. The mountains inspire me!",
        "assistant": "That's fascinating! What aspects of mountain landscapes do you find most captivating?",
        "knowledge": "Alice specializes in landscape painting and is particularly inspired by mountains.",
    }

    await memory.process_interaction(
        content=interaction2, tags=["art", "landscapes", "mountains"]
    )
    print("✓ Working Memory: Added new conversation details")
    print("✓ Episodic Memory: Stored the detailed interaction")
    print("✓ Semantic Memory: Learned about Alice's landscape specialization")

    print("\n=== 4a. Memory-Informed Response ===")
    print("The system uses its accumulated knowledge to engage more deeply...")

    # Get context for a more informed response
    context = memory.get_context_for_llm(
        knowledge_query="what do we know about Alice's art interests?",
    )
    print("\nAvailable Context for Response:")
    if "relevant_knowledge" in context:
        for knowledge in context["relevant_knowledge"]:
            print(f"- {knowledge['content']}")

    # Third interaction showing memory integration
    interaction3 = {
        "timestamp": datetime.now().isoformat(),
        "user": "Yes! I love how the light plays on the mountain peaks at sunset.",
        "assistant": "That's beautiful! Given your interest in capturing mountain landscapes and light effects, have you explored impressionist techniques?",
        "knowledge": "Alice is interested in capturing light effects in her mountain landscapes.",
    }

    await memory.process_interaction(
        content=interaction3, tags=["art", "technique", "light"]
    )
    print("\n✓ Memory Integration: Used previous knowledge to ask relevant follow-up")
    print("✓ Semantic Memory: Added new details about Alice's artistic interests")

    print("\n=== 4b. Connecting Past and Present ===")
    print("The system recalls earlier interactions to make connections...")

    # Fourth interaction demonstrating memory connections
    interaction4 = {
        "timestamp": datetime.now().isoformat(),
        "user": "Actually, yes! I started with traditional landscapes, but I've been experimenting with impressionist styles lately.",
        "assistant": "That's a wonderful progression! From our earlier conversation about your love of painting mountains, I can see how impressionism would help capture those fleeting light effects you mentioned.",
        "knowledge": [
            "Alice has progressed from traditional to impressionist landscape painting",
            "She combines mountain subjects with impressionist techniques",
        ],
    }

    await memory.process_interaction(
        content=interaction4, tags=["art", "technique", "progression"]
    )
    print(
        "\n✓ Episodic Memory: Connected earlier mountain discussion with new information"
    )
    print("✓ Semantic Memory: Built a richer understanding of Alice's artistic journey")

    print("\n=== 4c. Demonstrating Long-term Learning ===")
    print("The system shows how it builds a comprehensive understanding...")

    # Show accumulated knowledge
    context = memory.get_context_for_llm(
        knowledge_query="tell me about Alice's artistic journey and interests",
    )

    print("\nAccumulated Knowledge About Alice:")
    if "relevant_knowledge" in context:
        print("Artist Profile:")
        for knowledge in context["relevant_knowledge"]:
            print(f"- {knowledge['content']}")

    print(
        "\n✓ Knowledge Building: Demonstrated how individual interactions build a rich knowledge base"
    )

    print("\n=== 5. Remembering Previous Interactions ===")
    print("Later in the conversation, the system can recall what it learned...")

    # Get context from all memory systems
    context = memory.get_context_for_llm(
        include_episodes=2,
        knowledge_query="what do we know about Alice's art?",
        action_query="how to engage with user interests",
    )

    print("\nWhat the system remembers:")
    if "recent_episodes" in context:
        print("From Episodic Memory:")
        for episode in context["recent_episodes"]:
            if "content" in episode:
                content = episode["content"]
                if isinstance(content, dict):
                    print(f"- {content.get('user', '')}")

    if "relevant_knowledge" in context:
        print("\nFrom Semantic Memory:")
        for knowledge in context["relevant_knowledge"]:
            print(f"- {knowledge['content']}")

    if "relevant_procedures" in context:
        print("\nFrom Procedural Memory:")
        for proc in context["relevant_procedures"]:
            print(f"- Knows how to: {proc['name']}")

    print("\n=== 6. Learning from Success ===")
    print("The system updates its success rates based on the interaction...")

    # Process successful interaction
    success_record = {
        "timestamp": datetime.now().isoformat(),
        "action_sequence": {
            "name": "greet_and_engage",
            "success": True,
            "context": {
                "user_name": "Alice",
                "mentioned_interest": "painting",
            },
        },
    }

    await memory.process_interaction(
        content=success_record, tags=["successful_interaction"]
    )

    if procedures := memory.search_procedures(query="greeting procedure", limit=1):
        proc = procedures[0]
        print(f"✓ Success rate updated: {proc['success_rate']*100:.1f}%")
        print(f"✓ Times used: {proc['execution_count']}")

    print("\n=== 7. Memory Persistence ===")
    print("Starting a new session with the same memory storage...")

    # Create new memory system instance with same storage
    new_memory = CoALAMemorySystem(config)

    # Try to recall previous interactions
    context = new_memory.get_context_for_llm(
        include_episodes=2,
        knowledge_query="what do we know about Alice?",
    )

    print("\nWhat the new session remembers about Alice:")
    if "recent_episodes" in context:
        print("From Episodic Memory:")
        episodes = context["recent_episodes"]
        for episode in episodes:
            if "content" in episode:
                content = episode["content"]
                if isinstance(content, dict):
                    print(f"- {content.get('user', '')}")

    if "relevant_knowledge" in context:
        print("\nFrom Semantic Memory:")
        for knowledge in context["relevant_knowledge"]:
            print(f"- {knowledge['content']}")

    print("\n=== 8. Applying Knowledge in New Context ===")
    print(
        "A week later, the system uses its knowledge about Alice in a new situation..."
    )

    # Simulate a new context where knowledge about Alice is relevant
    art_class_interaction = {
        "timestamp": datetime.now().isoformat(),
        "user": "We're organizing a workshop on landscape painting techniques. Any recommendations?",
        "assistant": "I know an artist, Alice, who has an interesting journey from traditional to impressionist landscape painting, with a focus on mountain scenes and light effects. Her experience could be valuable for the workshop.",
        "context_query": "who do we know that could contribute to a landscape painting workshop?",
    }

    # Get relevant knowledge for the response
    workshop_context = new_memory.get_context_for_llm(
        knowledge_query="who has experience with landscape painting techniques?",
    )

    print("\nRelevant Knowledge for Workshop Planning:")
    if "relevant_knowledge" in workshop_context:
        print("Artist Recommendations:")
        for knowledge in workshop_context["relevant_knowledge"]:
            print(f"- {knowledge['content']}")

    print(
        "\n✓ Knowledge Application: Used past interactions to provide valuable recommendations"
    )
    print("✓ Context Integration: Connected Alice's experience to a new situation")

    print("\n=== Memory System Summary ===")
    print("Through this conversation, the CoALA system demonstrated:")
    print("1. Working Memory    - Kept track of the current conversation")
    print("2. Episodic Memory   - Remembered the specific interactions with Alice")
    print("3. Semantic Memory   - Built up knowledge about Alice's interests")
    print("4. Procedural Memory - Learned and improved its conversation skills")
    print("5. Memory Persistence- Maintained knowledge across sessions")
    print(
        "6. Knowledge Application- Used past interactions to provide valuable recommendations"
    )


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(simulate_conversation())
