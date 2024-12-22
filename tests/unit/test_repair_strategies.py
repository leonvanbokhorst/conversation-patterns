"""Unit tests for repair strategies pattern."""

import pytest
from typing import Dict, Any

from conversational_patterns.patterns.repair_strategies import (
    RepairStrategiesPattern,
    RepairStrategy,
)


@pytest.fixture
def repair_pattern() -> RepairStrategiesPattern:
    """Create a repair strategies pattern instance for testing."""
    return RepairStrategiesPattern()


@pytest.fixture
def sample_input() -> Dict[str, Any]:
    """Create sample input data for testing."""
    return {
        "utterance": "I want to know about the weather",
        "context": {"current_topic": "weather"},
        "confidence": 0.8,
        "repair_history": [],
    }


@pytest.mark.asyncio
async def test_process_no_repair_needed(repair_pattern: RepairStrategiesPattern, sample_input: Dict[str, Any]):
    """Test processing when no repair is needed."""
    result = await repair_pattern.process(sample_input)
    
    assert not result["needs_repair"]
    assert result["repair_strategy"] is None
    assert result["repair_response"] is None
    assert result["confidence"] == 0.8


@pytest.mark.asyncio
async def test_process_low_confidence(repair_pattern: RepairStrategiesPattern, sample_input: Dict[str, Any]):
    """Test processing with low confidence score."""
    sample_input["confidence"] = 0.3
    result = await repair_pattern.process(sample_input)
    
    assert result["needs_repair"]
    assert result["repair_strategy"] == RepairStrategy.CLARIFICATION.value
    assert "Could you rephrase that?" in result["repair_response"]
    assert result["confidence"] < 0.8


@pytest.mark.asyncio
async def test_process_context_inconsistency(repair_pattern: RepairStrategiesPattern, sample_input: Dict[str, Any]):
    """Test processing with context inconsistency."""
    sample_input["context"]["current_topic"] = "sports"
    result = await repair_pattern.process(sample_input)
    
    assert result["needs_repair"]
    assert result["repair_strategy"] == RepairStrategy.REFORMULATION.value
    assert "you're asking about sports?" in result["repair_response"]


@pytest.mark.asyncio
async def test_process_explicit_error(repair_pattern: RepairStrategiesPattern, sample_input: Dict[str, Any]):
    """Test processing with explicit error indicators."""
    sample_input["utterance"] = "I don't understand what you mean"
    result = await repair_pattern.process(sample_input)
    
    assert result["needs_repair"]
    assert result["repair_strategy"] in [s.value for s in RepairStrategy]
    assert result["repair_response"] is not None


@pytest.mark.asyncio
async def test_process_max_repairs_reached(repair_pattern: RepairStrategiesPattern, sample_input: Dict[str, Any]):
    """Test processing when maximum repair attempts are reached."""
    sample_input["confidence"] = 0.3
    sample_input["repair_history"] = ["clarification", "reformulation", "confirmation"]
    result = await repair_pattern.process(sample_input)
    
    assert result["needs_repair"]
    assert result["repair_strategy"] == RepairStrategy.CONFIRMATION.value
    assert "Is that correct?" in result["repair_response"]


@pytest.mark.asyncio
async def test_confidence_decay(repair_pattern: RepairStrategiesPattern, sample_input: Dict[str, Any]):
    """Test confidence score decay with repair attempts."""
    sample_input["confidence"] = 0.8
    sample_input["repair_history"] = ["clarification", "reformulation"]
    result = await repair_pattern.process(sample_input)
    
    assert result["confidence"] < 0.8


def test_error_indicators_detection(repair_pattern: RepairStrategiesPattern):
    """Test detection of error indicators in utterances."""
    error_utterances = [
        "What?",
        "I don't understand",
        "Could you repeat that?",
        "What do you mean?",
        "That's unclear",
        "I'm confused",
    ]
    
    for utterance in error_utterances:
        assert repair_pattern._has_error_indicators(utterance)


def test_context_consistency_check(repair_pattern: RepairStrategiesPattern):
    """Test context consistency checking."""
    context = {"current_topic": "weather", "pending_reference": "temperature"}
    
    # Test topic consistency
    assert repair_pattern._is_context_consistent(
        context, "What's the weather like today?"
    )
    assert not repair_pattern._is_context_consistent(
        context, "Who won the game yesterday?"
    )
    
    # Test reference resolution
    assert repair_pattern._is_context_consistent(
        context, "The temperature is 25 degrees"
    )
    assert not repair_pattern._is_context_consistent(
        context, "It's quite nice today"
    ) 