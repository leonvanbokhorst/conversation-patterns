"""Configuration for the CoALA memory system."""

from typing import Optional
from pydantic import BaseModel


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""

    redis_url: str
    working_memory_max_tokens: int = 1000
    summary_model: Optional[str] = "gpt-4o-mini"  # Default to gpt-4o-mini
    episodic_namespace: str = "episodes"
    semantic_persist_dir: str = "./semantic_memory"
    procedural_persist_dir: str = "./procedural_memory"
    embedding_model: Optional[str] = "text-embedding-3-small"
