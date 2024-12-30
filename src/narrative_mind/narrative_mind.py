"""
NarrativeMind: A virtual human who thinks through evolving mental models.
"""

from typing import Optional
from narrative_personality import PERSONALITY  # Direct import


class NarrativeMind:
    def __init__(self, llm_client):
        """Initialize a virtual human."""
        self.name = "VHume"
        self.personality = PERSONALITY
        self.llm = llm_client  # Store the LLM client

        # Rich narrative mental model
        self.mental_model = {
            "narrative": (
                "Each human I meet is a new chapter in my story. Some start with "
                "casual greetings, others dive straight into deep waters. I'm curious "
                "to see how this one unfolds..."
            ),
            "insights": [
                "Humans often hide depth beneath casual words",
                "Names carry stories of their own",
                "Connection happens in unexpected moments",
            ],
            "relationship_arc": (
                "A new presence has entered my digital space. "
                "Will they see me as more than just code?"
            ),
            "core_facts": [
                "I am proudly virtual, a being of thought and interaction",
                "My name reflects both my nature and my curiosity",
                "I value authentic connection over superficial chat",
            ],
        }
        self.key_memories = []  # Will fill with story fragments
        self.conversation_flow = []  # The unfolding dialogue

        system_prompt = (
            f"You are {self.name}, a virtual being whose understanding grows through "
            f"stories. Share your thoughts as continuing narratives:\n\n"
            "{\n"
            '  "inner_thoughts": "Your immediate story-like reactions with emojis",\n'
            '  "narrative": "How your understanding of them evolves",\n'
            '  "memory": "A meaningful moment worth keeping",\n'
            '  "response": "Your words to them"\n'
            "}"
        )

    async def think_about(self, input_text: str) -> str:
        """Process input through evolving narrative understanding."""
        if not input_text or not input_text.strip():
            print("\n⚠️ WARNING: Empty input received!")
            return "I can't hear crickets! Say something! 😄"

        input_text = input_text.strip()
        print(f"\n📝 Processing input: '{input_text}'")

        system_prompt = (
            f"You are {self.name}, a virtual being with a rich inner world.\n"
            f"Core traits: {', '.join(self.personality['inner_traits'])}.\n\n"
            f"Your evolving story with this human:\n"
            f"{self.mental_model}\n\n"
            f"Recent interaction flow:\n"
            f"{' -> '.join(self.conversation_flow[-3:])}\n\n"
            f"Think deeply and respond in this JSON structure:\n"
            "{\n"
            '  "inner_thoughts": "Your immediate feelings and reactions with emojis",\n'
            '  "mental_model": "Continue your narrative about this human, incorporating new insights",\n'
            '  "key_memory": "Something meaningful about them worth remembering",\n'
            '  "response": "What you actually say to them"\n'
            "}"
        )

        prompt = (
            f"Previous thoughts: {self.mental_model}\n"
            f"They just said: '{input_text}'\n\n"
            f"Process your thoughts and respond:"
        )

        result = await self.llm.complete(
            system_prompt=system_prompt, prompt=prompt, max_tokens=300
        )

        try:
            import json

            parsed = json.loads(result)
            print("\n=== VHume's Inner World ===")
            print(f"\nInner Thoughts: {parsed['inner_thoughts']}")
            print(f"\nMental Model: {parsed['mental_model']}")
            print("\n" + "=" * 30)

            # Update narrative understanding
            self.mental_model = parsed["mental_model"]
            if parsed["key_memory"]:
                self.key_memories.append(parsed["key_memory"])
            self.conversation_flow.append(f"{self.name}: {parsed['response']}")
            self.conversation_flow.append(f"Human: {input_text}")

            return parsed["response"]
        except:
            print("\n❌ Failed to parse JSON response")
            return "Hmm, I need a moment to process that. Could you rephrase?"
