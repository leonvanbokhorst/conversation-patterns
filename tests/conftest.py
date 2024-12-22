"""
Shared test fixtures for the conversational patterns test suite.
"""

import pytest
from typing import Dict, Any

from conversational_patterns.config.settings import (
    SystemConfig,
    TurnTakingConfig,
    ContextConfig,
    ResponseConfig,
    RepairConfig,
)


@pytest.fixture
def base_config() -> SystemConfig:
    """Create a base system configuration for testing."""
    return SystemConfig(
        turn_taking=TurnTakingConfig(
            min_response_delay=0.1, max_response_delay=0.3, interruption_threshold=0.5
        ),
        context=ContextConfig(
            max_history_turns=5, context_decay_rate=0.2, min_context_relevance=0.4
        ),
        response=ResponseConfig(
            variation_threshold=0.6,
            style_consistency_weight=0.4,
            context_adaptation_rate=0.2,
        ),
        repair=RepairConfig(
            error_detection_threshold=0.7,
            max_repair_attempts=2,
            repair_strategy_weights={
                "clarification": 0.5,
                "reformulation": 0.3,
                "confirmation": 0.2,
            },
        ),
        debug_mode=True,
        log_level="DEBUG",
    )


@pytest.fixture
def sample_conversation_turn() -> Dict[str, Any]:
    """Create a sample conversation turn for testing."""
    return {
        "speaker": "user",
        "utterance": "Hello, how are you?",
        "timestamp": 1234567890,
        "metadata": {"confidence": 0.95, "sentiment": "positive", "intent": "greeting"},
    }


@pytest.fixture
def conversation_history() -> list[Dict[str, Any]]:
    """Create a sample conversation history for testing."""
    return [
        {
            "speaker": "user",
            "utterance": "Hi there!",
            "timestamp": 1234567880,
            "metadata": {"confidence": 0.98, "intent": "greeting"},
        },
        {
            "speaker": "system",
            "utterance": "Hello! How can I help you today?",
            "timestamp": 1234567885,
            "metadata": {"confidence": 1.0, "intent": "greeting_response"},
        },
        {
            "speaker": "user",
            "utterance": "I have a question about...",
            "timestamp": 1234567890,
            "metadata": {"confidence": 0.92, "intent": "question"},
        },
    ]
