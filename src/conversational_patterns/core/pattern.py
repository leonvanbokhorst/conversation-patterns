"""
Base interface for conversational patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ConversationState(BaseModel):
    """Represents the current state of a conversation."""

    turn_count: int = 0
    current_speaker: Optional[str] = None
    last_utterance: Optional[str] = None
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class Pattern(ABC):
    """Base interface for all conversational patterns."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the pattern with configuration.

        Args:
            config: Pattern-specific configuration dictionary
        """
        self.config = config
        self.state = ConversationState()

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data according to the pattern.

        Args:
            input_data: Dictionary containing input data for pattern processing

        Returns:
            Dictionary containing processed output data
        """
        pass

    @abstractmethod
    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update the pattern's internal state.

        Args:
            new_state: Dictionary containing state updates
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the pattern to its initial state."""
        pass

    @property
    @abstractmethod
    def pattern_type(self) -> str:
        """Return the type of the pattern."""
        pass
