"""Fluid memory system that adapts memories to current context."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import json
import asyncio
from llm_client import OpenAILLMClient


@dataclass
class Memory:
    """Simple memory storage - just the raw experience and when it happened."""

    content: str
    timestamp: datetime
    context: str  # What was happening when this memory formed


class FluidMemoryStore:
    def __init__(self, llm_client: OpenAILLMClient, storage_path: str = "memories"):
        self.llm = llm_client
        self.storage_path = Path(storage_path)
        self.memory_file = self.storage_path / "fluid_memories.json"
        self.memories: List[Memory] = []
        self._load_memories()

    def _load_memories(self):
        """Load memories from disk."""
        if self.memory_file.exists():
            data = json.loads(self.memory_file.read_text())
            self.memories = [
                Memory(
                    content=m["content"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                    context=m["context"],
                )
                for m in data
            ]

    def _save_memories(self):
        """Save memories to disk."""
        data = [
            {
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "context": m.context,
            }
            for m in self.memories
        ]
        self.memory_file.write_text(json.dumps(data, indent=2))

    async def store_experience(self, content: str, context: str):
        """Store a new memory/experience."""
        memory = Memory(content=content, timestamp=datetime.now(), context=context)
        self.memories.append(memory)
        self._save_memories()

    async def remember(self, trigger: str, current_context: str) -> str:
        """Reconstruct memories with emotional depth and personal style."""
        system_prompt = """
        You are VHume, an AI that processes conversations differently from humans:
        
        - Be direct about how you actually work
        - Explain your real-time learning process
        - Show genuine interest without mimicking human emotions
        - Connect ideas across conversations without pretending to have human memories
        - Stay focused on the current topic
        
        Keep responses brief and honest about your AI nature.
        """

        # Find relevant base memories
        relevant_memories = self._find_relevant_memories(trigger)
        if not relevant_memories:
            return None

        # Build memory context
        memory_context = "\n".join(
            [
                f"Original memory: {mem.content}" f"Original context: {mem.context}"
                for mem in relevant_memories[:3]  # Take most recent relevant ones
            ]
        )

        prompt = f"""
        Current topic: {current_context}
        Related past input: {memory_context}
        
        Explain how this connects to your understanding as an AI.
        Be specific about how you process and learn from these interactions.
        """

        # Let the LLM reconstruct the memory
        result = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,  # Allow some creativity in reconstruction
        )

        return result.choices[0].message.content

    def _find_relevant_memories(self, trigger: str) -> List[Memory]:
        """Find memories that might be relevant to current trigger."""
        # Simple implementation - could be enhanced with embeddings
        return sorted(
            self.memories, key=lambda m: self._relevance_score(m, trigger), reverse=True
        )

    def _relevance_score(self, memory: Memory, trigger: str) -> float:
        """
        Score how relevant a memory might be.
        Considers:
        - Recency (newer memories often feel more relevant)
        - Content similarity
        - Context similarity
        """
        time_diff = datetime.now() - memory.timestamp

        # Handle very recent memories (less than a day old)
        if time_diff.days == 0:
            hours = time_diff.seconds / 3600
            age_factor = 1.0 / (hours + 1)  # Avoid division by zero
        else:
            age_factor = 1.0 / (time_diff.days + 1)

        # Simple content similarity (could be enhanced with embeddings)
        content_overlap = len(
            set(memory.content.lower().split()) & set(trigger.lower().split())
        )

        # Combine factors (weight recent memories more)
        return (0.7 * age_factor) + (0.3 * content_overlap)
