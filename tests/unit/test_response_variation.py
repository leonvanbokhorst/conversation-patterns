"""
Unit tests for the response variation pattern.
"""

import pytest
from typing import Dict, Any, List

from conversational_patterns.patterns.response_variation import ResponseVariationPattern
from conversational_patterns.config.settings import ResponseConfig


@pytest.fixture
def pattern() -> ResponseVariationPattern:
    """Create a response variation pattern instance for testing."""
    config = {
        "variation_threshold": 0.3,
        "style_consistency_weight": 0.5,
        "context_adaptation_rate": 0.2,
    }
    return ResponseVariationPattern(config)


@pytest.fixture
def sample_responses() -> List[str]:
    """Create sample response options for testing."""
    return [
        "I would be happy to assist you with that.",
        "Let me help you with that.",
        "Sure, I can help!",
        "No problem, I'll take care of it.",
        "I'll help you right away.",
    ]


@pytest.mark.asyncio
async def test_pattern_initialization(pattern: ResponseVariationPattern):
    """Test pattern initialization."""
    assert pattern.pattern_type == "response_variation"
    assert pattern.state.turn_count == 0
    assert pattern.state.last_utterance is None


@pytest.mark.asyncio
async def test_response_selection(pattern: ResponseVariationPattern, sample_responses):
    """Test basic response selection."""
    input_data = {
        "response_options": sample_responses,
        "context": {},
        "style": {"formality": 0.5, "complexity": 0.5},
    }

    result = await pattern.process(input_data)

    assert "selected_response" in result
    assert "variation_score" in result
    assert "style_score" in result
    assert result["selected_response"] in sample_responses
    assert 0 <= result["variation_score"] <= 1
    assert 0 <= result["style_score"] <= 1


@pytest.mark.asyncio
async def test_variation_scoring(pattern: ResponseVariationPattern):
    """Test variation score calculations."""
    # First response
    input_data1 = {
        "response_options": ["Hello, how can I help you today?"],
        "context": {},
        "style": {},
    }
    await pattern.process(input_data1)

    # Similar response should have low variation score
    similar_response = "Hello, how may I assist you today?"
    variation_score = pattern._calculate_variation(similar_response)
    assert variation_score < 0.5

    # Different response should have high variation score
    different_response = "The weather is nice today."
    variation_score = pattern._calculate_variation(different_response)
    assert variation_score > 0.8


@pytest.mark.asyncio
async def test_style_consistency(pattern: ResponseVariationPattern):
    """Test style consistency calculations."""
    formal_response = "I would be delighted to assist you with your inquiry."
    informal_response = "Hey, sure thing! I'll help ya out!"

    # Test formal style matching
    formal_score = pattern._calculate_style_score(
        formal_response, {"formality": 0.9, "complexity": 0.7}
    )

    # Test informal style matching
    informal_score = pattern._calculate_style_score(
        informal_response, {"formality": 0.1, "complexity": 0.3}
    )

    assert formal_score > 0.7
    assert informal_score > 0.7


@pytest.mark.asyncio
async def test_context_adaptation(pattern: ResponseVariationPattern, sample_responses):
    """Test context-based response adaptation."""
    # Test with consistency requirement
    consistent_input = {
        "response_options": sample_responses,
        "context": {"requires_consistency": True},
        "style": {},
    }
    consistent_result = await pattern.process(consistent_input)

    # Test with creativity allowance
    creative_input = {
        "response_options": sample_responses,
        "context": {"allows_creativity": True},
        "style": {},
    }
    creative_result = await pattern.process(creative_input)

    # Context should influence variation scores
    assert consistent_result["variation_score"] <= creative_result["variation_score"]


@pytest.mark.asyncio
async def test_formality_measurement(pattern: ResponseVariationPattern):
    """Test formality level measurement."""
    formal_text = "Would you please assist me with this matter? Thank you."
    informal_text = "Hey there! Yeah, I can help ya with that!"
    neutral_text = "This is a simple statement."

    formal_score = pattern._measure_formality(formal_text)
    informal_score = pattern._measure_formality(informal_text)
    neutral_score = pattern._measure_formality(neutral_text)

    assert formal_score > 0.7
    assert informal_score < 0.3
    assert 0.4 <= neutral_score <= 0.6


@pytest.mark.asyncio
async def test_complexity_measurement(pattern: ResponseVariationPattern):
    """Test text complexity measurement."""
    simple_text = "This is a very simple text."
    complex_text = "The implementation demonstrates sophisticated algorithmic patterns."

    simple_score = pattern._measure_complexity(simple_text)
    complex_score = pattern._measure_complexity(complex_text)

    assert simple_score < complex_score
    assert 0 <= simple_score <= 1
    assert 0 <= complex_score <= 1


@pytest.mark.asyncio
async def test_empty_inputs(pattern: ResponseVariationPattern):
    """Test handling of empty inputs."""
    input_data = {"response_options": [], "context": {}, "style": {}}

    result = await pattern.process(input_data)

    assert result["selected_response"] == ""
    assert result["variation_score"] == 0
    assert result["style_score"] == 1


@pytest.mark.asyncio
async def test_reset_functionality(pattern: ResponseVariationPattern):
    """Test pattern reset functionality."""
    input_data = {
        "response_options": ["Test response"],
        "context": {"test": True},
        "style": {},
    }

    await pattern.process(input_data)
    pattern.reset()

    assert pattern.state.turn_count == 0
    assert pattern.state.last_utterance is None
    assert pattern.state.context == {}
