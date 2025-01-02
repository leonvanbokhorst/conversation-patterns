"""Episodic memory component for the CoALA memory system."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid

from pydantic import BaseModel, Field
from langchain_core.memory import BaseMemory
import redis


class Episode(BaseModel):
    """Model for an episodic memory entry."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    content: Dict[str, Any]
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class CoALAEpisodicMemory(BaseMemory, BaseModel):
    """
    Episodic memory implementation following CoALA framework.
    Uses Redis for persistent storage of episodes.
    """

    redis_url: str
    namespace: str = "episodes"
    summary_model: Optional[str] = None
    _redis: Optional[redis.Redis] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._redis = redis.Redis.from_url(self.redis_url, decode_responses=True)

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables used in this memory object."""
        return ["recent_episodes"]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation turn to memory."""
        # Create a new episode from the interaction
        episode = Episode(content={"inputs": inputs, "outputs": outputs})

        # Store in Redis
        self._store_episode(episode)

    def _store_episode(self, episode: Episode) -> None:
        """Store an episode in Redis."""
        key = f"{self.namespace}:{episode.id}"
        self._redis.set(key, episode.model_dump_json())

        # Add to sorted set for timestamp-based retrieval
        self._redis.zadd(
            f"{self.namespace}:timeline",
            {episode.id: datetime.fromisoformat(episode.timestamp).timestamp()},
        )

        # Add to sets for tag-based retrieval
        for tag in episode.tags:
            self._redis.sadd(f"{self.namespace}:tag:{tag}", episode.id)

    def store_episode(self, content: Dict[str, Any], tags: List[str] = None) -> str:
        """Store a new episode in memory."""
        episode = Episode(content=content, tags=tags or [])
        self._store_episode(episode)
        return episode.id

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve a specific episode by ID."""
        key = f"{self.namespace}:{episode_id}"
        data = self._redis.get(key)
        return Episode.model_validate_json(data) if data else None

    def get_recent_episodes(self, limit: int = 5) -> List[Episode]:
        """Get the most recent episodes."""
        # Get recent episode IDs from the sorted set
        episode_ids = self._redis.zrevrange(f"{self.namespace}:timeline", 0, limit - 1)

        return self._retrieve_full_episodes(episode_ids)

    def search_by_tags(self, tags: List[str], match_all: bool = True) -> List[Episode]:
        """
        Search episodes by tags.

        Args:
            tags: List of tags to search for
            match_all: If True, episode must have all tags. If False, any tag matches.
        """
        if not tags:
            return []

        # Get episode IDs that match tags
        tag_keys = [f"{self.namespace}:tag:{tag}" for tag in tags]
        if match_all:
            episode_ids = self._redis.sinter(tag_keys)
        else:
            episode_ids = self._redis.sunion(tag_keys)

        return self._retrieve_full_episodes(episode_ids)

    def _retrieve_full_episodes(self, episode_ids):
        """Retrieve full episodes from Redis."""
        episodes = []
        for episode_id in episode_ids:
            if episode := self.get_episode(episode_id):
                episodes.append(episode)
        return episodes

    def clear(self) -> None:
        """Clear all episodic memory."""
        # Get all keys in the namespace
        pattern = f"{self.namespace}:*"
        if keys := self._redis.keys(pattern):
            self._redis.delete(*keys)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for LLM context."""
        # Get recent episodes
        recent = self.get_recent_episodes(limit=3)
        return {
            "recent_episodes": [
                {
                    "timestamp": episode.timestamp,
                    "content": episode.content,
                    "summary": episode.summary,
                }
                for episode in recent
            ]
        }
