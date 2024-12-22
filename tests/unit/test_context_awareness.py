"""
Unit tests for the context awareness pattern.
"""

import pytest
from typing import Dict, Any

from conversational_patterns.patterns.context_awareness import ContextAwarenessPattern
from conversational_patterns.config.settings import ContextConfig


@pytest.fixture
def pattern() -> ContextAwarenessPattern:
    """Create a context awareness pattern instance for testing."""
    config = {
        "max_history_turns": 5,
        "context_decay_rate": 0.2,
        "min_context_relevance": 0.4,
    }
    return ContextAwarenessPattern(config)


@pytest.fixture
def sample_history() -> list[Dict[str, Any]]:
    """Create a sample conversation history for testing."""
    return [
        {
            "turn": 1,
            "utterance": "Hi there!",
            "metadata": {
                "confidence": 0.9,
                "intent": "greeting",
                "topic": "introduction",
            },
        },
        {
            "turn": 2,
            "utterance": "I need help with my account",
            "metadata": {
                "confidence": 0.8,
                "intent": "help_request",
                "topic": "account",
            },
        },
        {
            "turn": 3,
            "utterance": "I forgot my password",
            "metadata": {
                "confidence": 0.95,
                "intent": "password_reset",
                "topic": "account",
            },
        },
    ]


@pytest.mark.asyncio
async def test_pattern_initialization(pattern: ContextAwarenessPattern):
    """Test pattern initialization."""
    assert pattern.pattern_type == "context_awareness"
    assert pattern.state.turn_count == 0
    assert pattern.state.context == {}


@pytest.mark.asyncio
async def test_context_processing(pattern: ContextAwarenessPattern, sample_history):
    """Test basic context processing."""
    input_data = {
        "utterance": "Can you help me reset it?",
        "metadata": {"confidence": 0.85, "intent": "confirm_reset", "topic": "account"},
        "history": sample_history,
    }

    result = await pattern.process(input_data)

    assert "relevant_context" in result
    assert "context_score" in result
    assert "suggested_topics" in result
    assert len(result["relevant_context"]) > 0
    assert 0 <= result["context_score"] <= 1
    assert "account" in result["suggested_topics"]


@pytest.mark.asyncio
async def test_context_relevance(pattern: ContextAwarenessPattern, sample_history):
    """Test context relevance calculations."""
    input_data = {
        "utterance": "Test utterance",
        "metadata": {"confidence": 1.0},
        "history": sample_history,
    }

    result = await pattern.process(input_data)

    # Check that more recent items have higher relevance
    relevance_scores = [item.get("relevance", 0) for item in result["relevant_context"]]

    # Relevance should decrease with age
    for i in range(len(relevance_scores) - 1):
        assert relevance_scores[i] >= relevance_scores[i + 1]


@pytest.mark.asyncio
async def test_context_decay(pattern: ContextAwarenessPattern):
    """Test context decay over turns."""
    initial_context = {"confidence": 1.0, "importance": 1.0}

    # Process multiple turns to test decay
    for _ in range(5):
        input_data = {
            "utterance": "Test",
            "metadata": {"confidence": 0.5},
            "history": [],
        }
        await pattern.process(input_data)

        # Context values should decay
        for value in pattern.state.context.values():
            if isinstance(value, (int, float)):
                assert value <= 1.0


@pytest.mark.asyncio
async def test_topic_identification(pattern: ContextAwarenessPattern, sample_history):
    """Test topic identification from context."""
    input_data = {
        "utterance": "I need to change my password",
        "metadata": {
            "confidence": 0.9,
            "intent": "password_change",
            "topic": "account",
        },
        "history": sample_history,
    }

    result = await pattern.process(input_data)

    # Should identify topics from history and current utterance
    assert "account" in result["suggested_topics"]
    assert "password_change" in result["suggested_topics"]


@pytest.mark.asyncio
async def test_context_merging(pattern: ContextAwarenessPattern):
    """Test merging of new context with existing context."""
    # First update
    input_data1 = {
        "utterance": "First utterance",
        "metadata": {"confidence": 0.8, "topic": "topic1"},
        "history": [],
    }
    await pattern.process(input_data1)

    # Second update with overlapping context
    input_data2 = {
        "utterance": "Second utterance",
        "metadata": {"confidence": 0.9, "topic": "topic2"},
        "history": [],
    }
    result = await pattern.process(input_data2)

    # Check that context was properly merged
    assert pattern.state.context.get("confidence") == 0.9
    assert "topic1" in str(pattern.state.context)
    assert "topic2" in str(pattern.state.context)


@pytest.mark.asyncio
async def test_reset_functionality(pattern: ContextAwarenessPattern, sample_history):
    """Test pattern reset functionality."""
    input_data = {
        "utterance": "Test utterance",
        "metadata": {"confidence": 1.0},
        "history": sample_history,
    }

    await pattern.process(input_data)
    pattern.reset()

    assert pattern.state.turn_count == 0
    assert pattern.state.last_utterance is None
    assert pattern.state.context == {}
