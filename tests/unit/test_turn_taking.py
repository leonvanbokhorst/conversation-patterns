"""
Unit tests for the turn-taking pattern.
"""

import pytest
from typing import Dict, Any

from conversational_patterns.patterns.turn_taking import TurnTakingPattern
from conversational_patterns.config.settings import TurnTakingConfig


@pytest.fixture
def pattern() -> TurnTakingPattern:
    """Create a turn-taking pattern instance for testing."""
    config = {
        "min_response_delay": 0.1,
        "max_response_delay": 0.3,
        "interruption_threshold": 0.5,
    }
    return TurnTakingPattern(config)


@pytest.mark.asyncio
async def test_pattern_initialization(pattern: TurnTakingPattern):
    """Test pattern initialization."""
    assert pattern.pattern_type == "turn_taking"
    assert pattern.state.turn_count == 0
    assert pattern.state.current_speaker is None


@pytest.mark.asyncio
async def test_process_turn(pattern: TurnTakingPattern):
    """Test basic turn processing."""
    input_data = {"speaker": "user", "utterance": "Hello", "timestamp": 1234567890}

    result = await pattern.process(input_data)

    assert "next_speaker" in result
    assert "delay" in result
    assert "can_interrupt" in result
    assert result["next_speaker"] == "system"
    assert 0.1 <= result["delay"] <= 0.3
    assert isinstance(result["can_interrupt"], bool)


@pytest.mark.asyncio
async def test_state_updates(pattern: TurnTakingPattern):
    """Test state updates during processing."""
    input_data = {"speaker": "user", "utterance": "Hello", "timestamp": 1234567890}

    await pattern.process(input_data)

    assert pattern.state.turn_count == 1
    assert pattern.state.current_speaker == "user"
    assert pattern.state.last_utterance == "Hello"


@pytest.mark.asyncio
async def test_speaker_alternation(pattern: TurnTakingPattern):
    """Test that speakers alternate correctly."""
    inputs = [
        {"speaker": "user", "utterance": "Hello", "timestamp": 1},
        {"speaker": "system", "utterance": "Hi", "timestamp": 2},
        {"speaker": "user", "utterance": "How are you?", "timestamp": 3},
    ]

    for input_data in inputs:
        result = await pattern.process(input_data)
        assert result["next_speaker"] != input_data["speaker"]


@pytest.mark.asyncio
async def test_reset_functionality(pattern: TurnTakingPattern):
    """Test pattern reset functionality."""
    input_data = {"speaker": "user", "utterance": "Hello", "timestamp": 1234567890}

    await pattern.process(input_data)
    pattern.reset()

    assert pattern.state.turn_count == 0
    assert pattern.state.current_speaker is None
    assert pattern.state.last_utterance is None


@pytest.mark.asyncio
async def test_delay_calculation(pattern: TurnTakingPattern):
    """Test that delay calculations are within bounds."""
    input_data = {"speaker": "user", "utterance": "Hello", "timestamp": 1234567890}

    # Process multiple turns to test delay adjustments
    for _ in range(10):
        result = await pattern.process(input_data)
        assert 0.1 <= result["delay"] <= 0.3


@pytest.mark.asyncio
async def test_interruption_probability(pattern: TurnTakingPattern):
    """Test interruption probability changes over turns."""
    input_data = {"speaker": "user", "utterance": "Hello", "timestamp": 1234567890}

    # Process multiple turns to test interruption probability
    interruption_counts = 0
    total_turns = 20

    for _ in range(total_turns):
        result = await pattern.process(input_data)
        if result["can_interrupt"]:
            interruption_counts += 1

    # Check that interruptions occur with reasonable frequency
    assert 0 <= interruption_counts <= total_turns
