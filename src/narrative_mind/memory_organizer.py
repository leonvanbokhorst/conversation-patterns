"""Memory organization and consolidation for narrative mind."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import difflib


@dataclass
class ConsolidatedMemory:
    """A memory that might combine several related memories."""

    content: str
    memory_type: str
    first_seen: datetime
    last_seen: datetime
    frequency: int
    themes: List[str]
    emotional_context: Dict[str, float]
    related_memories: List[str]  # IDs of constituent memories
    importance_score: float


class MemoryOrganizer:
    """Handles memory consolidation and organization."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def should_consolidate(self, memory1: str, memory2: str) -> bool:
        """Check if two memories are similar enough to consolidate."""
        similarity = difflib.SequenceMatcher(None, memory1, memory2).ratio()
        return similarity > self.similarity_threshold

    def consolidate_memories(self, memories: List[dict]) -> List[ConsolidatedMemory]:
        """Consolidate similar memories into unified memories."""
        consolidated = []
        processed = set()

        for i, mem1 in enumerate(memories):
            if i in processed:
                continue

            related = []
            themes = set(mem1["themes"])
            emotions = mem1["emotional_context"].copy()

            # Find similar memories
            for j, mem2 in enumerate(memories[i + 1 :], i + 1):
                if self.should_consolidate(mem1["content"], mem2["content"]):
                    related.append(mem2["id"])
                    processed.add(j)
                    themes.update(mem2["themes"])
                    for emotion, value in mem2["emotional_context"].items():
                        if emotion in emotions:
                            emotions[emotion] = (emotions[emotion] + value) / 2
                        else:
                            emotions[emotion] = value

            # Create consolidated memory
            consolidated.append(
                ConsolidatedMemory(
                    content=mem1["content"],
                    memory_type=mem1["memory_type"],
                    first_seen=datetime.fromisoformat(mem1["timestamp"]),
                    last_seen=datetime.now(),
                    frequency=len(related) + 1,
                    themes=list(themes),
                    emotional_context=emotions,
                    related_memories=related,
                    importance_score=self._calculate_importance(
                        len(related), len(themes)
                    ),
                )
            )

        return consolidated

    def _calculate_importance(self, frequency: int, theme_count: int) -> float:
        """Calculate memory importance based on frequency and theme coverage."""
        return (0.7 * frequency + 0.3 * theme_count) / (frequency + theme_count)
