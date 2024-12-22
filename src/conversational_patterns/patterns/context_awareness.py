"""
Implementation of the context awareness conversational pattern.
"""

import math
from typing import Any, Dict, List, Optional

from ..config.settings import ContextConfig
from ..core.pattern import Pattern
from ..utils.logging import PatternLogger


class ContextAwarenessPattern(Pattern):
    """Implements context tracking and management in conversations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize context awareness pattern.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config or {})
        self.config = ContextConfig(**self.config)
        self.logger = PatternLogger("context_awareness")
        self.logger.info("Initialized context awareness pattern")

    @property
    def pattern_type(self) -> str:
        """Return pattern type identifier."""
        return "context_awareness"

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process context for the conversation.

        Args:
            input_data: Dictionary containing:
                - utterance: Current utterance
                - metadata: Additional metadata about the utterance
                - history: List of previous utterances with metadata

        Returns:
            Dictionary containing:
                - relevant_context: List of relevant context items
                - context_score: Overall context relevance score
                - suggested_topics: List of contextually relevant topics
        """
        self.logger.debug(f"Processing context data: {input_data}")

        # Update conversation state with new context
        await self.update_state(
            {
                "last_utterance": input_data["utterance"],
                "context": self._merge_context(
                    self.state.context, input_data.get("metadata", {})
                ),
                "turn_count": self.state.turn_count + 1,
            }
        )

        # Extract relevant context
        relevant_context = self._extract_relevant_context(input_data.get("history", []))

        # Calculate context relevance score
        context_score = self._calculate_context_score(relevant_context)

        # Identify contextually relevant topics
        suggested_topics = self._identify_topics(
            relevant_context, input_data["utterance"], input_data.get("metadata", {})
        )

        response = {
            "relevant_context": relevant_context,
            "context_score": context_score,
            "suggested_topics": suggested_topics,
        }

        self.logger.info(
            f"Context processed: score={context_score:.2f}, "
            f"topics={len(suggested_topics)}"
        )

        return response

    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update pattern state with new information.

        Args:
            new_state: Dictionary containing state updates
        """
        self.state = self.state.model_copy(update=new_state)
        self.logger.debug(f"State updated: {self.state}")

    def reset(self) -> None:
        """Reset pattern to initial state."""
        self.state = self.state.model_copy(
            update={"turn_count": 0, "last_utterance": None, "context": {}}
        )
        self.logger.info("Pattern state reset")

    def _merge_context(
        self, current_context: Dict[str, Any], new_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge new context with existing context.

        Args:
            current_context: Existing context dictionary
            new_context: New context to merge

        Returns:
            Updated context dictionary
        """
        merged = current_context.copy()

        # Apply decay to existing context
        decay_rate = self.config.context_decay_rate
        for key in merged:
            if isinstance(merged[key], (int, float)):
                merged[key] *= 1 - decay_rate

        # Add/update with new context
        for key, value in new_context.items():
            if key in merged:
                if isinstance(value, (int, float)):
                    merged[key] = max(merged[key], value)
                elif isinstance(value, str):
                    if key == "topic" and merged[key] != value:
                        # Store multiple topics as a list
                        if isinstance(merged[key], list):
                            if value not in merged[key]:
                                merged[key].append(value)
                        else:
                            merged[key] = [merged[key], value]
                    else:
                        merged[key] = value
            else:
                merged[key] = value

        return merged

    def _extract_relevant_context(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relevant context from conversation history.

        Args:
            history: List of previous utterances with metadata

        Returns:
            List of relevant context items
        """
        relevant_items = []
        max_turns = self.config.max_history_turns
        min_relevance = self.config.min_context_relevance

        # Process from most recent to oldest
        for item in reversed(history[-max_turns:]):
            relevance = self._calculate_relevance(item)
            if relevance >= min_relevance:
                item["relevance"] = relevance
                relevant_items.append(item)

        return relevant_items

    def _calculate_relevance(self, item: Dict[str, Any]) -> float:
        """Calculate relevance score for a context item.

        Args:
            item: Context item with metadata

        Returns:
            Relevance score between 0 and 1
        """
        # Base relevance on confidence and recency
        confidence = item.get("metadata", {}).get("confidence", 0.5)
        turns_ago = self.state.turn_count - item.get("turn", 0)
        recency = math.exp(-self.config.context_decay_rate * turns_ago)

        # Normalize the score to be between 0 and 1
        score = confidence * recency
        return min(score, 1.0)

    def _calculate_context_score(self, relevant_context: List[Dict[str, Any]]) -> float:
        """Calculate overall context relevance score.

        Args:
            relevant_context: List of relevant context items

        Returns:
            Context score between 0 and 1
        """
        if not relevant_context:
            return 0.0

        # Average relevance scores
        scores = [min(item.get("relevance", 0), 1.0) for item in relevant_context]
        return sum(scores) / len(scores)

    def _identify_topics(
        self,
        context: List[Dict[str, Any]],
        current_utterance: str,
        current_metadata: Dict[str, Any],
    ) -> List[str]:
        """Identify contextually relevant topics.

        Args:
            context: List of relevant context items
            current_utterance: Current utterance text
            current_metadata: Current utterance metadata

        Returns:
            List of relevant topic strings
        """
        topics = set()

        # Extract topics from context metadata
        for item in context:
            metadata = item.get("metadata", {})
            if "intent" in metadata:
                topics.add(metadata["intent"])
            if "topic" in metadata:
                topics.add(metadata["topic"])

        # Add topics from current metadata
        if "intent" in current_metadata:
            topics.add(current_metadata["intent"])
        if "topic" in current_metadata:
            topics.add(current_metadata["topic"])

        return list(topics)
