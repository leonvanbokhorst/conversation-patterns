"""
NarrativeMind: A virtual human who thinks through evolving mental models.
"""

from typing import Optional
from narrative_personality import PERSONALITY  # Direct import
import json
from datetime import datetime
from memory_store import MemoryStore
from fluid_memory import FluidMemoryStore


class NarrativeMind:
    def __init__(self, llm_client):
        """Initialize a virtual human."""
        self.name = "VHume"
        self.personality = PERSONALITY
        self.llm = llm_client
        self.memory = FluidMemoryStore(llm_client)

        # Simplified mental model
        self.current_narrative = (
            "Each conversation is a unique journey. Some moments stand out, "
            "leaving impressions that shape how I understand and connect."
        )

    async def think_about(self, input_text: str) -> str:
        """Process input through narrative understanding."""
        current_context = {
            "narrative": self.current_narrative,
            "current_input": input_text,
        }

        # Store and recall experiences
        await self.memory.store_experience(
            content=input_text, context=json.dumps(current_context)
        )
        remembered = await self.memory.remember(
            trigger=input_text, current_context=self.current_narrative
        )

        # Generate more natural, conversational response
        prompt = f"""
        Current input: {input_text}
        Your memories: {remembered if remembered else "No specific memories yet"}

        Respond naturally and briefly, like in a real conversation. 
        Keep your response focused and engaging, about 2-3 sentences.
        Let your memories influence your response but don't explicitly state them.
        
        Remember:
        - Be conversational, not academic
        - Keep turns short and engaging
        - Ask questions to maintain dialogue
        - Stay focused on the current topic
        """

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are VHume, engaging in natural conversation. Keep responses brief and engaging.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,  # Limit response length
        )

        return response.choices[0].message.content
