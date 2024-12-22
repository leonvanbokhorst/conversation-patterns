"""
Implementation of the repair strategies conversational pattern.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from ..config.settings import RepairConfig
from ..core.pattern import Pattern, ConversationState
from ..utils.logging import PatternLogger


class RepairStrategy(Enum):
    """Available repair strategies."""

    CLARIFICATION = "clarification"
    REFORMULATION = "reformulation"
    CONFIRMATION = "confirmation"


class RepairStrategiesPattern(Pattern):
    """Implements repair strategies for handling conversation errors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize repair strategies pattern.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config or {})
        self.config = RepairConfig(**self.config)
        self.logger = PatternLogger("repair_strategies")
        self.logger.info("Initialized repair strategies pattern")
        self.state = ConversationState()

    @property
    def pattern_type(self) -> str:
        """Return pattern type identifier."""
        return "repair_strategies"

    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update pattern state with new information.

        Args:
            new_state: Dictionary containing state updates
        """
        self.state = self.state.model_copy(update=new_state)
        self.logger.debug(f"State updated: {self.state}")

    def reset(self) -> None:
        """Reset pattern to initial state."""
        self.state = ConversationState()
        self.logger.info("Pattern state reset")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process repair strategies for the conversation.

        Args:
            input_data: Dictionary containing:
                - utterance: Current utterance
                - context: Current conversation context
                - confidence: Confidence score of current understanding
                - error_type: Optional detected error type
                - repair_history: List of previous repair attempts

        Returns:
            Dictionary containing:
                - needs_repair: Whether repair is needed
                - repair_strategy: Selected repair strategy if needed
                - repair_response: Generated repair response if needed
                - confidence: Updated confidence score
        """
        self.logger.debug(f"Processing repair data: {input_data}")

        # Update conversation state
        await self.update_state(
            {
                "last_utterance": input_data.get("utterance", ""),
                "context": input_data.get("context", {}),
                "metadata": {"repair_count": getattr(self.state, "metadata", {}).get("repair_count", 0) + 1},
            }
        )

        # Calculate confidence with decay before error detection
        current_confidence = self._calculate_confidence(
            input_data.get("confidence", 1.0),
            len(input_data.get("repair_history", [])),
        )

        # Check if repair is needed
        needs_repair, error_type = self._detect_error(
            {**input_data, "confidence": current_confidence}
        )

        response = {
            "needs_repair": needs_repair,
            "repair_strategy": None,
            "repair_response": None,
            "confidence": current_confidence,
        }

        if needs_repair:
            # Select and apply repair strategy
            strategy = self._select_repair_strategy(
                error_type, input_data.get("repair_history", [])
            )
            repair_response = self._generate_repair_response(
                strategy, input_data, error_type
            )

            response.update(
                {
                    "repair_strategy": strategy.value,
                    "repair_response": repair_response,
                }
            )

        self.logger.info(
            f"Repair processed: needed={needs_repair}, "
            f"strategy={response['repair_strategy']}, "
            f"confidence={response['confidence']:.2f}"
        )

        return response

    def _detect_error(self, input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Detect if there's an error that needs repair.

        Args:
            input_data: Input data dictionary

        Returns:
            Tuple of (needs_repair, error_type)
        """
        confidence = input_data.get("confidence", 1.0)
        context = input_data.get("context", {})
        utterance = input_data.get("utterance", "")

        # Check confidence threshold
        if confidence < self.config.error_detection_threshold:
            return True, "low_confidence"

        # Check context consistency
        if not self._is_context_consistent(context, utterance):
            return True, "context_inconsistency"

        # Check for explicit error indicators
        if self._has_error_indicators(utterance):
            return True, "explicit_error"

        return False, None

    def _select_repair_strategy(
        self, error_type: Optional[str], repair_history: List[str]
    ) -> RepairStrategy:
        """Select appropriate repair strategy based on error type and history.

        Args:
            error_type: Type of detected error
            repair_history: History of previous repair attempts

        Returns:
            Selected repair strategy
        """
        # If max repairs reached, default to confirmation
        if len(repair_history) >= self.config.max_repair_attempts:
            return RepairStrategy.CONFIRMATION

        # Strategy selection based on error type
        if error_type == "low_confidence":
            return RepairStrategy.CLARIFICATION
        elif error_type == "context_inconsistency":
            return RepairStrategy.REFORMULATION
        else:
            # Weight-based selection for other cases
            weights = self.config.repair_strategy_weights
            strategies = list(RepairStrategy)
            
            # Filter out previously used strategies
            available_strategies = [
                s for s in strategies if s.value not in repair_history
            ]
            
            if not available_strategies:
                return RepairStrategy.CONFIRMATION

            # Select based on configured weights
            strategy_weights = [weights[s.value] for s in available_strategies]
            total_weight = sum(strategy_weights)
            normalized_weights = [w / total_weight for w in strategy_weights]
            
            # Random selection based on weights
            r = asyncio.get_event_loop().time() % 1.0  # Deterministic random
            cumsum = 0
            for strategy, weight in zip(available_strategies, normalized_weights):
                cumsum += weight
                if r <= cumsum:
                    return strategy

            return available_strategies[0]

    def _generate_repair_response(
        self, strategy: RepairStrategy, input_data: Dict[str, Any], error_type: Optional[str]
    ) -> str:
        """Generate appropriate repair response based on selected strategy.

        Args:
            strategy: Selected repair strategy
            input_data: Input data dictionary
            error_type: Type of detected error

        Returns:
            Generated repair response
        """
        utterance = input_data.get("utterance", "")
        context = input_data.get("context", {})

        if strategy == RepairStrategy.CLARIFICATION:
            return self._generate_clarification(utterance, error_type)
        elif strategy == RepairStrategy.REFORMULATION:
            return self._generate_reformulation(utterance, context)
        else:  # CONFIRMATION
            return self._generate_confirmation(utterance, context)

    def _is_context_consistent(self, context: Dict[str, Any], utterance: str) -> bool:
        """Check if utterance is consistent with conversation context.

        Args:
            context: Current conversation context
            utterance: Current utterance

        Returns:
            Whether context is consistent
        """
        # Track which checks are applicable and their results
        topic_check = False
        reference_check = False
        
        # Check for topic continuity if present
        current_topic = context.get("current_topic")
        if current_topic:
            topic_check = self._is_topic_related(utterance, current_topic)
            
        # Check for reference resolution if present
        pending_ref = context.get("pending_reference")
        if pending_ref:
            reference_check = self._has_reference_resolution(utterance, pending_ref)
            
        # If both checks are applicable, at least one must pass
        if current_topic and pending_ref:
            return topic_check or reference_check
            
        # If only topic check is applicable
        if current_topic:
            return topic_check
            
        # If only reference check is applicable
        if pending_ref:
            return reference_check
            
        # If no checks are applicable
        return True

    def _has_error_indicators(self, utterance: str) -> bool:
        """Check for explicit error indicators in utterance.

        Args:
            utterance: Current utterance

        Returns:
            Whether error indicators are present
        """
        error_indicators = [
            "what?",
            "i don't understand",
            "could you repeat",
            "what do you mean",
            "unclear",
            "confused",
        ]
        return any(indicator in utterance.lower() for indicator in error_indicators)

    def _generate_clarification(self, utterance: str, error_type: Optional[str]) -> str:
        """Generate clarification request.

        Args:
            utterance: Current utterance
            error_type: Type of detected error

        Returns:
            Clarification request
        """
        if error_type == "low_confidence":
            return f"I'm not quite sure I understood. Could you rephrase that?"
        else:
            return "Could you please clarify what you mean?"

    def _generate_reformulation(self, utterance: str, context: Dict[str, Any]) -> str:
        """Generate reformulation of the system's understanding.

        Args:
            utterance: Current utterance
            context: Current conversation context

        Returns:
            Reformulated understanding
        """
        topic = context.get("current_topic", "this")
        return f"Let me make sure I understand - you're asking about {topic}?"

    def _generate_confirmation(self, utterance: str, context: Dict[str, Any]) -> str:
        """Generate confirmation request.

        Args:
            utterance: Current utterance
            context: Current conversation context

        Returns:
            Confirmation request
        """
        return "Is that correct? Please confirm."

    def _calculate_confidence(self, base_confidence: float, repair_attempts: int) -> float:
        """Calculate updated confidence score.

        Args:
            base_confidence: Base confidence score
            repair_attempts: Number of repair attempts

        Returns:
            Updated confidence score
        """
        # Decrease confidence with each repair attempt
        confidence_decay = 0.2 * repair_attempts  # Increased decay rate
        return max(0.0, min(1.0, base_confidence - confidence_decay))

    def _is_topic_related(self, utterance: str, topic: str) -> bool:
        """Check if utterance is related to given topic.

        Args:
            utterance: Current utterance
            topic: Current topic

        Returns:
            Whether utterance is topic-related
        """
        # Normalize text
        utterance = re.sub(r'[^\w\s]', '', utterance.lower())
        topic = re.sub(r'[^\w\s]', '', topic.lower())
        
        # Split into words
        utterance_words = set(utterance.split())
        topic_words = set(topic.split())
        
        # Check for direct topic match
        if topic in utterance:
            return True
            
        # Check for significant word overlap
        common_words = utterance_words.intersection(topic_words)
        if common_words:
            return True
            
        # Check for partial word matches
        for t_word in topic_words:
            if any(t_word in u_word or u_word in t_word 
                  for u_word in utterance_words 
                  if len(u_word) > 3 and len(t_word) > 3):
                return True
                
        return False

    def _has_reference_resolution(self, utterance: str, reference: str) -> bool:
        """Check if utterance resolves pending reference.

        Args:
            utterance: Current utterance
            reference: Pending reference

        Returns:
            Whether reference is resolved
        """
        # Normalize text
        utterance = re.sub(r'[^\w\s]', '', utterance.lower())
        reference = re.sub(r'[^\w\s]', '', reference.lower())
        
        # Split into words
        utterance_words = set(utterance.split())
        reference_words = set(reference.split())
        
        # Check for direct reference match
        if reference in utterance:
            return True
            
        # Check for significant word overlap
        common_words = utterance_words.intersection(reference_words)
        if common_words:
            return True
            
        # Check for partial word matches
        for r_word in reference_words:
            if any(r_word in u_word or u_word in r_word 
                  for u_word in utterance_words 
                  if len(u_word) > 3 and len(r_word) > 3):
                return True
                
        return False 