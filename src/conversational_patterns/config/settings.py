"""
Configuration settings for the conversational patterns system.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TurnTakingConfig(BaseModel):
    """Configuration for turn-taking pattern."""

    min_response_delay: float = Field(
        0.5, description="Minimum delay before response in seconds"
    )
    max_response_delay: float = Field(
        2.0, description="Maximum delay before response in seconds"
    )
    interruption_threshold: float = Field(
        0.8, description="Threshold for interruption probability"
    )


class ContextConfig(BaseModel):
    """Configuration for context awareness pattern."""

    max_history_turns: int = Field(
        10, description="Maximum number of conversation turns to keep"
    )
    context_decay_rate: float = Field(
        0.1, description="Rate at which context importance decays"
    )
    min_context_relevance: float = Field(
        0.3, description="Minimum relevance score to maintain context"
    )


class ResponseConfig(BaseModel):
    """Configuration for response variation pattern."""

    variation_threshold: float = Field(
        0.7, description="Threshold for response variation"
    )
    style_consistency_weight: float = Field(
        0.5, description="Weight for maintaining style consistency"
    )
    context_adaptation_rate: float = Field(
        0.3, description="Rate of adaptation to context"
    )


class RepairConfig(BaseModel):
    """Configuration for repair strategies pattern."""

    error_detection_threshold: float = Field(
        0.6, description="Threshold for error detection"
    )
    max_repair_attempts: int = Field(3, description="Maximum number of repair attempts")
    repair_strategy_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "clarification": 0.4,
            "reformulation": 0.3,
            "confirmation": 0.3,
        }
    )


class SystemConfig(BaseModel):
    """Main system configuration."""

    turn_taking: TurnTakingConfig = Field(default_factory=TurnTakingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    debug_mode: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Logging level")


# Default configuration instance
default_config = SystemConfig()
