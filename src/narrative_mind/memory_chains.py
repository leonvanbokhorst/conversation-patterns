"""Memory chain management for narrative understanding."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import json


@dataclass
class MemoryNode:
    """A single memory node in the chain."""

    content: str
    memory_type: str
    timestamp: str
    context: dict
    themes: List[str]
    connections: List[str]  # IDs of related memories
    emotional_context: Dict[str, float]  # emotion -> intensity


class MemoryChain:
    """Manages temporal chains of related memories."""

    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}  # id -> node
        self.themes: Dict[str, List[str]] = {}  # theme -> [memory_ids]
        self.timeline: List[str] = []  # Ordered memory IDs

    def add_memory(
        self,
        content: str,
        memory_type: str,
        context: dict,
        themes: List[str] = None,
        emotional_context: Dict[str, float] = None,
    ) -> str:
        """Add a new memory to the chain."""
        memory_id = f"mem_{datetime.now().isoformat()}"

        # Create node
        node = MemoryNode(
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now().isoformat(),
            context=context,
            themes=themes or [],
            connections=[],
            emotional_context=emotional_context or {},
        )

        # Store node
        self.nodes[memory_id] = node
        self.timeline.append(memory_id)

        # Update theme indices
        for theme in node.themes:
            if theme not in self.themes:
                self.themes[theme] = []
            self.themes[theme].append(memory_id)

        # Find and create connections
        self._update_connections(memory_id)

        return memory_id

    def _update_connections(self, memory_id: str):
        """Find and update memory connections."""
        current = self.nodes[memory_id]

        # Connect to memories with shared themes
        for theme in current.themes:
            for other_id in self.themes.get(theme, []):
                if other_id != memory_id:
                    current.connections.append(other_id)
                    self.nodes[other_id].connections.append(memory_id)

    def get_recent_context(self, limit: int = 5) -> List[MemoryNode]:
        """Get most recent memories."""
        recent_ids = self.timeline[-limit:]
        return [self.nodes[mid] for mid in recent_ids]

    def get_themed_memories(self, theme: str, limit: int = 5) -> List[MemoryNode]:
        """Get memories related to a theme."""
        memory_ids = self.themes.get(theme, [])
        return [self.nodes[mid] for mid in memory_ids[-limit:]]

    def get_connected_memories(self, memory_id: str) -> List[MemoryNode]:
        """Get memories connected to a specific memory."""
        node = self.nodes.get(memory_id)
        if not node:
            return []
        return [self.nodes[mid] for mid in node.connections]

    def save(self, path: str):
        """Save memory chain to file."""
        data = {
            "nodes": {k: vars(v) for k, v in self.nodes.items()},
            "themes": self.themes,
            "timeline": self.timeline,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MemoryChain":
        """Load memory chain from file."""
        chain = cls()
        with open(path, "r") as f:
            data = json.load(f)

        chain.themes = data["themes"]
        chain.timeline = data["timeline"]
        chain.nodes = {k: MemoryNode(**v) for k, v in data["nodes"].items()}

        return chain
