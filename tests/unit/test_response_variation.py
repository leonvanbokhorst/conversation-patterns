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


@pytest.mark.asyncio
async def test_personality_matching(pattern: ResponseVariationPattern):
    """Test Big Five personality trait matching."""
    # Test high openness and extraversion
    open_extraverted = (
        "I'm excited to explore innovative solutions with you! Let's discover new "
        "approaches together and experiment with unique ideas."
    )
    open_extraverted_style = {
        "personality": {
            "openness": 0.8,
            "conscientiousness": 0.5,
            "extraversion": 0.8,
            "agreeableness": 0.5,
            "neuroticism": 0.3,
        }
    }
    open_extraverted_score = pattern._calculate_style_score(
        open_extraverted, open_extraverted_style
    )

    # Test high conscientiousness and low neuroticism
    conscientious_stable = (
        "I will systematically analyze the requirements and prepare a detailed, "
        "structured plan. You can be confident in our methodical approach."
    )
    conscientious_stable_style = {
        "personality": {
            "openness": 0.5,
            "conscientiousness": 0.8,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.2,
        }
    }
    conscientious_stable_score = pattern._calculate_style_score(
        conscientious_stable, conscientious_stable_style
    )

    # Test high agreeableness and moderate neuroticism
    agreeable_moderate = (
        "I understand your concerns and I'm happy to help find a gentle solution. "
        "Let's work together supportively to address any uncertainties."
    )
    agreeable_moderate_style = {
        "personality": {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.8,
            "neuroticism": 0.5,
        }
    }
    agreeable_moderate_score = pattern._calculate_style_score(
        agreeable_moderate, agreeable_moderate_style
    )

    # Verify scores
    assert open_extraverted_score > 0.7
    assert conscientious_stable_score > 0.7
    assert agreeable_moderate_score > 0.7

    # Test trait consistency
    assert pattern._measure_personality_match(
        open_extraverted, {"openness": 0.8, "extraversion": 0.8}
    ) > pattern._measure_personality_match(
        open_extraverted, {"openness": 0.2, "extraversion": 0.2}
    )


@pytest.mark.asyncio
async def test_individual_traits(pattern: ResponseVariationPattern):
    """Test individual Big Five trait calculations."""
    # Test openness
    high_openness = "Let's explore innovative and creative approaches to understand this fascinating challenge."
    low_openness = (
        "We'll stick to our proven, conventional methods that are well-established."
    )

    # Test conscientiousness
    high_conscientiousness = (
        "I'll prepare a detailed, systematic plan with thorough documentation."
    )
    low_conscientiousness = "Let's keep it flexible and adaptable, going with the flow."

    # Test extraversion
    high_extraversion = (
        "I'm excited to engage in this dynamic, interactive session with you!"
    )
    low_extraversion = "I'll quietly reflect on this and provide a measured response."

    # Test agreeableness
    high_agreeableness = (
        "I'm happy to help and will be patient and supportive throughout the process."
    )
    low_agreeableness = "Here are the objective facts and straightforward analysis."

    # Test neuroticism
    high_neuroticism = (
        "I'm a bit worried about potential issues, so let's proceed carefully."
    )
    low_neuroticism = (
        "I'm confident we can handle this with a steady, balanced approach."
    )

    # Test each dimension
    traits_to_test = [
        ("openness", high_openness, low_openness),
        ("conscientiousness", high_conscientiousness, low_conscientiousness),
        ("extraversion", high_extraversion, low_extraversion),
        ("agreeableness", high_agreeableness, low_agreeableness),
        ("neuroticism", high_neuroticism, low_neuroticism),
    ]

    for trait, high_text, low_text in traits_to_test:
        # Test high value of trait
        high_score = pattern._calculate_style_score(
            high_text, {"personality": {trait: 0.8}}
        )
        # Test low value of trait
        low_score = pattern._calculate_style_score(
            low_text, {"personality": {trait: 0.2}}
        )

        assert high_score > 0.6
        assert low_score > 0.6


@pytest.mark.asyncio
async def test_personality_integration(pattern: ResponseVariationPattern):
    """Test integration of Big Five personality traits with other style aspects."""
    input_data = {
        "response_options": [
            "I'm excited to explore innovative solutions with you!",  # High openness/extraversion
            "Let me analyze this systematically and prepare a detailed plan.",  # High conscientiousness
            "I understand your concerns and I'm happy to help find a solution together.",  # High agreeableness
        ],
        "context": {"allows_creativity": True},
        "style": {
            "formality": 0.7,
            "complexity": 0.6,
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.4,
                "extraversion": 0.7,
                "agreeableness": 0.6,
                "neuroticism": 0.3,
            },
        },
    }

    result = await pattern.process(input_data)

    assert "selected_response" in result
    assert "variation_score" in result
    assert "style_score" in result
    assert result["style_score"] > 0

    # Process with contrasting personality
    input_data["style"]["personality"] = {
        "openness": 0.3,
        "conscientiousness": 0.9,  # Strongly different from first personality
        "extraversion": 0.3,
        "agreeableness": 0.5,
        "neuroticism": 0.4,
    }

    different_result = await pattern.process(input_data)

    # Different personality should select different response
    assert result["selected_response"] != different_result["selected_response"]

    # The second response should be more systematic/conscientious
    assert (
        "systematic" in different_result["selected_response"].lower()
        or "detailed" in different_result["selected_response"].lower()
    )


@pytest.mark.asyncio
async def test_personality_edge_cases(pattern: ResponseVariationPattern):
    """Test edge cases in personality trait matching."""
    # Test neutral text
    neutral_text = "This is a simple statement without personality indicators."
    neutral_style = {
        "personality": {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
    }
    neutral_score = pattern._calculate_style_score(neutral_text, neutral_style)
    assert 0.4 <= neutral_score <= 0.8  # Should be moderate

    # Test mixed traits
    mixed_text = (
        "While I'm excited to explore new ideas, let's proceed carefully "
        "with a structured plan to address any concerns systematically."
    )
    mixed_style = {
        "personality": {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.6,
            "agreeableness": 0.5,
            "neuroticism": 0.6,
        }
    }
    mixed_score = pattern._calculate_style_score(mixed_text, mixed_style)
    assert mixed_score > 0.6  # Should handle mixed traits well

    # Test empty text
    empty_score = pattern._calculate_style_score("", neutral_style)
    assert empty_score == 1.0  # Should return default score
