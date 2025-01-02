"""Main coordinator for the CoALA memory system."""

from typing import Dict, List, Optional, Any, Callable

from working_memory import CoALAWorkingMemory
from episodic_memory import CoALAEpisodicMemory, Episode
from semantic_memory import CoALASemanticMemory
from procedural_memory import CoALAProceduralMemory, ActionStep
from config import MemoryConfig


class CoALAMemorySystem:
    """
    Central coordinator for the CoALA memory system.

    Manages and coordinates between different memory types:
    - Working memory for current context
    - Episodic memory for experiences
    - Semantic memory for knowledge
    - Procedural memory for skills
    """

    def __init__(self, config: MemoryConfig):
        """Initialize the memory system with configuration."""
        self.config = config

        # Initialize memory components
        self.working_memory = CoALAWorkingMemory(
            max_tokens=config.working_memory_max_tokens
        )

        self.episodic_memory = CoALAEpisodicMemory(
            redis_url=config.redis_url,
            summary_model=config.summary_model,
            namespace=config.episodic_namespace,
        )

        self.semantic_memory = CoALASemanticMemory(
            persist_directory=config.semantic_persist_dir,
            embedding_model=config.embedding_model,
        )

        self.procedural_memory = CoALAProceduralMemory(
            persist_directory=config.procedural_persist_dir,
            embedding_model=config.embedding_model,
        )

    def register_action(self, name: str, func: Callable) -> None:
        """Register an action function that can be used in procedural memory."""
        self.procedural_memory.register_function(name, func)

    async def process_interaction(
        self, content: Dict[str, Any], tags: List[str] = None
    ) -> str:
        """
        Process a new interaction through the memory system.

        Args:
            content: Interaction content
            tags: Optional tags for episodic memory

        Returns:
            episode_id: ID of stored episode
        """
        # Update working memory
        self.working_memory.add_context("last_interaction", content)

        # Store in episodic memory
        episode_id = self.episodic_memory.store_episode(content=content, tags=tags)

        # Extract and store knowledge if present
        if "knowledge" in content:
            self.semantic_memory.store_knowledge(
                content=content["knowledge"],
                metadata={"source": "interaction", "episode_id": episode_id},
            )

        # Extract and store action sequence if present
        if "action_sequence" in content:
            sequence = content["action_sequence"]
            if isinstance(sequence, dict):
                if "name" in sequence and "steps" not in sequence:
                    # This is a procedure execution result
                    # TODO: Update success rate and execution count
                    pass
                else:
                    # This is a new procedure definition
                    self.procedural_memory.store_sequence(
                        name=sequence["name"],
                        description=sequence["description"],
                        steps=[ActionStep(**step) for step in sequence["steps"]],
                        context=sequence.get("context"),
                        tags=(
                            sequence.get("tags", []) + tags
                            if tags
                            else sequence.get("tags", [])
                        ),
                    )

        return episode_id

    def get_context_for_llm(
        self,
        include_episodes: int = 3,
        knowledge_query: str = "",
        action_query: str = "",
        min_success_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get formatted context for LLM prompting.

        Args:
            include_episodes: Number of recent episodes to include
            knowledge_query: Optional query for semantic memory
            action_query: Optional query for procedural memory
            min_success_rate: Optional minimum success rate for action sequences

        Returns:
            Dict containing context from all memory types
        """
        # Get working memory context
        context = self.working_memory.load_memory_variables({})

        # Add recent episodes
        recent_episodes = self.episodic_memory.get_recent_episodes(include_episodes)
        context["recent_episodes"] = [
            {"timestamp": episode.timestamp, "summary": episode.summary}
            for episode in recent_episodes
        ]

        # Add relevant knowledge if query provided
        if knowledge_query:
            knowledge = self.semantic_memory.load_memory_variables(
                {"query": knowledge_query}
            )
            context["relevant_knowledge"] = knowledge["relevant_knowledge"]

        # Add relevant procedures if query provided
        if action_query:
            procedures = self.procedural_memory.load_memory_variables(
                {"query": action_query, "min_success_rate": min_success_rate}
            )
            context["available_actions"] = procedures["available_actions"]
            context["relevant_procedures"] = procedures["relevant_procedures"]

        return context

    def search_episodes(self, tags: List[str], match_all: bool = True) -> List[Episode]:
        """Search episodic memory by tags."""
        return self.episodic_memory.search_by_tags(tags, match_all)

    def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search semantic memory for relevant knowledge."""
        return self.semantic_memory.search_knowledge(query, limit, metadata_filter)

    def search_procedures(
        self,
        query: str,
        limit: int = 5,
        min_success_rate: Optional[float] = None,
        required_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search procedural memory for relevant action sequences."""
        return self.procedural_memory.search_sequences(
            query, limit, min_success_rate, required_tags
        )

    async def execute_procedure(
        self, sequence_id: str, context: Dict[str, Any] = None
    ) -> bool:
        """Execute an action sequence from procedural memory."""
        return await self.procedural_memory.execute_sequence(sequence_id, context)

    def clear_all(self) -> None:
        """Clear all memory systems."""
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.procedural_memory.clear()
