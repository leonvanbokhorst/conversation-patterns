"""
Implementation of the turn-taking conversational pattern.
"""

import asyncio
import random
from typing import Any, Dict, Optional

from ..config.settings import TurnTakingConfig
from ..core.pattern import Pattern
from ..utils.logging import PatternLogger


class TurnTakingPattern(Pattern):
    """Implements turn-taking behavior in conversations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize turn-taking pattern.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config or {})
        self.config = TurnTakingConfig(**self.config)
        self.logger = PatternLogger("turn_taking")
        self.logger.info("Initialized turn-taking pattern")

    @property
    def pattern_type(self) -> str:
        """Return pattern type identifier."""
        return "turn_taking"

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process turn-taking for the conversation.

        Args:
            input_data: Dictionary containing:
                - speaker: Current speaker identifier
                - utterance: Current utterance
                - timestamp: Utterance timestamp

        Returns:
            Dictionary containing:
                - next_speaker: Next speaker identifier
                - delay: Suggested delay before next turn
                - can_interrupt: Whether interruption is allowed
        """
        self.logger.debug(f"Processing turn data: {input_data}")

        # Update conversation state
        await self.update_state(
            {
                "current_speaker": input_data["speaker"],
                "last_utterance": input_data["utterance"],
                "turn_count": self.state.turn_count + 1,
            }
        )

        # Calculate response delay
        delay = self._calculate_delay()

        # Determine if interruption is allowed
        can_interrupt = self._check_interruption()

        # Determine next speaker
        next_speaker = self._select_next_speaker(input_data["speaker"])

        response = {
            "next_speaker": next_speaker,
            "delay": delay,
            "can_interrupt": can_interrupt,
        }

        self.logger.info(
            f"Turn processed: next={next_speaker}, "
            f"delay={delay:.2f}, interrupt={can_interrupt}"
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
            update={"turn_count": 0, "current_speaker": None, "last_utterance": None}
        )
        self.logger.info("Pattern state reset")

    def _calculate_delay(self) -> float:
        """Calculate appropriate delay before next turn.

        Returns:
            Delay in seconds
        """
        min_delay = self.config.min_response_delay
        max_delay = self.config.max_response_delay

        # Add some natural variation to the delay
        delay = random.uniform(min_delay, max_delay)

        # Adjust based on conversation state
        if self.state.turn_count > 5:  # More fluid after initial turns
            delay = max(min_delay, delay * 0.8)

        return delay

    def _check_interruption(self) -> bool:
        """Determine if interruption should be allowed.

        Returns:
            Boolean indicating if interruption is allowed
        """
        threshold = self.config.interruption_threshold

        # Base probability on threshold
        base_prob = random.random()

        # Allow more interruptions in established conversations
        if self.state.turn_count > 10:
            base_prob *= 1.2

        return base_prob > threshold

    def _select_next_speaker(self, current_speaker: str) -> str:
        """Select the next speaker based on conversation state.

        Args:
            current_speaker: Current speaker identifier

        Returns:
            Next speaker identifier
        """
        # Simple alternating pattern - should be enhanced for multi-party
        return "system" if current_speaker == "user" else "user"
