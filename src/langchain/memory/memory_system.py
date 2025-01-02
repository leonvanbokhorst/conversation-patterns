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

        # Initialize memory components with error handling
        try:
            # Initialize working memory (in-memory, should always work)
            self.working_memory = CoALAWorkingMemory(
                max_tokens=config.working_memory_max_tokens
            )

            # Initialize episodic memory (requires Redis)
            try:
                self.episodic_memory = CoALAEpisodicMemory(
                    redis_url=config.redis_url,
                    summary_model=config.summary_model,
                    namespace=config.episodic_namespace,
                )
            except Exception as e:
                print(f"\n⚠️  Warning: Episodic memory initialization failed: {str(e)}")
                print("The system will continue without episodic memory capabilities.")
                self.episodic_memory = None

            # Initialize semantic memory (requires filesystem)
            try:
                self.semantic_memory = CoALASemanticMemory(
                    persist_directory=config.semantic_persist_dir,
                    embedding_model=config.embedding_model,
                )
            except Exception as e:
                print(f"\n⚠️  Warning: Semantic memory initialization failed: {str(e)}")
                print("The system will continue without semantic memory capabilities.")
                self.semantic_memory = None

            # Initialize procedural memory (requires filesystem)
            try:
                self.procedural_memory = CoALAProceduralMemory(
                    persist_directory=config.procedural_persist_dir,
                    embedding_model=config.embedding_model,
                )
            except Exception as e:
                print(
                    f"\n⚠️  Warning: Procedural memory initialization failed: {str(e)}"
                )
                print(
                    "The system will continue without procedural memory capabilities."
                )
                self.procedural_memory = None

        except Exception as e:
            raise RuntimeError(f"Failed to initialize memory system: {str(e)}") from e

    def _check_subsystem(self, name: str, subsystem: Any) -> None:
        """Check if a subsystem is available."""
        if subsystem is None:
            raise RuntimeError(
                f"{name} is not available. Check initialization warnings for details."
            )

    def register_action(self, name: str, func: Callable) -> None:
        """Register an action function that can be used in procedural memory."""
        self.procedural_memory.register_function(name, func)

    async def process_interaction(
        self, content: Dict[str, Any], tags: List[str] = None
    ) -> Optional[str]:
        """
        Process a new interaction through the memory system.

        Args:
            content: Interaction content
            tags: Optional tags for episodic memory

        Returns:
            episode_id: ID of stored episode, or None if episodic memory is unavailable
        """
        # Update working memory
        self.working_memory.add_context("last_interaction", content)

        # Store in episodic memory if available
        episode_id = None
        if self.episodic_memory:
            try:
                episode_id = self.episodic_memory.store_episode(
                    content=content, tags=tags
                )
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to store episode: {str(e)}")

        # Extract and store knowledge if present and semantic memory is available
        if self.semantic_memory and "knowledge" in content:
            try:
                self.semantic_memory.store_knowledge(
                    content=content["knowledge"],
                    metadata={"source": "interaction", "episode_id": episode_id},
                )
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to store knowledge: {str(e)}")

        # Extract and store action sequence if present and procedural memory is available
        if self.procedural_memory and "action_sequence" in content:
            try:
                sequence = content["action_sequence"]
                if isinstance(sequence, dict):
                    if "name" in sequence and "steps" not in sequence:
                        # This is a procedure execution result
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
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to store procedure: {str(e)}")

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

        # Add recent episodes if available
        if self.episodic_memory:
            try:
                recent_episodes = self.episodic_memory.get_recent_episodes(
                    include_episodes
                )
                context["recent_episodes"] = [
                    {"timestamp": episode.timestamp, "content": episode.content}
                    for episode in recent_episodes
                ]
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to retrieve episodes: {str(e)}")
                context["recent_episodes"] = []

        # Add relevant knowledge if query provided and semantic memory available
        if knowledge_query and self.semantic_memory:
            try:
                knowledge = self.semantic_memory.load_memory_variables(
                    {"query": knowledge_query}
                )
                context["relevant_knowledge"] = knowledge["relevant_knowledge"]
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to retrieve knowledge: {str(e)}")
                context["relevant_knowledge"] = []

        # Add relevant procedures if query provided and procedural memory available
        if action_query and self.procedural_memory:
            try:
                procedures = self.procedural_memory.load_memory_variables(
                    {"query": action_query, "min_success_rate": min_success_rate}
                )
                context["available_actions"] = procedures["available_actions"]
                context["relevant_procedures"] = procedures["relevant_procedures"]
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to retrieve procedures: {str(e)}")
                context["available_actions"] = []
                context["relevant_procedures"] = []

        return context

    def search_episodes(self, tags: List[str], match_all: bool = True) -> List[Episode]:
        """Search episodic memory by tags."""
        if not self.episodic_memory:
            print("\n⚠️  Warning: Episodic memory is not available")
            return []
        try:
            return self.episodic_memory.search_by_tags(tags, match_all)
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to search episodes: {str(e)}")
            return []

    def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search semantic memory for relevant knowledge."""
        if not self.semantic_memory:
            print("\n⚠️  Warning: Semantic memory is not available")
            return []
        try:
            return self.semantic_memory.search_knowledge(query, limit, metadata_filter)
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to search knowledge: {str(e)}")
            return []

    def search_procedures(
        self,
        query: str,
        limit: int = 5,
        min_success_rate: Optional[float] = None,
        required_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search procedural memory for relevant action sequences."""
        if not self.procedural_memory:
            print("\n⚠️  Warning: Procedural memory is not available")
            return []
        try:
            return self.procedural_memory.search_sequences(
                query, limit, min_success_rate, required_tags
            )
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to search procedures: {str(e)}")
            return []

    async def execute_procedure(
        self, sequence_id: str, context: Dict[str, Any] = None
    ) -> bool:
        """Execute an action sequence from procedural memory."""
        if not self.procedural_memory:
            print("\n⚠️  Warning: Procedural memory is not available")
            return False
        try:
            return await self.procedural_memory.execute_sequence(sequence_id, context)
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to execute procedure: {str(e)}")
            return False

    def clear_all(self) -> None:
        """Clear all memory systems."""
        self.working_memory.clear()

        if self.episodic_memory:
            try:
                self.episodic_memory.clear()
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to clear episodic memory: {str(e)}")

        if self.semantic_memory:
            try:
                self.semantic_memory.clear()
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to clear semantic memory: {str(e)}")

        if self.procedural_memory:
            try:
                self.procedural_memory.clear()
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to clear procedural memory: {str(e)}")
