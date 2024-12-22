"""
Logging utilities for the conversational patterns system.
"""

import logging
import sys
from typing import Optional

from ..config.settings import default_config


def setup_logger(
    name: str, level: Optional[str] = None, format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with the specified configuration.

    Args:
        name: Name of the logger
        level: Logging level (defaults to config value)
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Use default config if level not specified
    if level is None:
        level = default_config.log_level

    # Set numeric level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric_level)

    # Create console handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        # Use custom or default format
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(pattern_type)s - %(message)s"
            )

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class PatternLogger:
    """Logger wrapper with pattern-specific context."""

    def __init__(self, pattern_type: str):
        """Initialize pattern logger.

        Args:
            pattern_type: Type of pattern this logger is for
        """
        self.logger = setup_logger(f"pattern.{pattern_type}")
        self.pattern_type = pattern_type

    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal logging method with pattern context.

        Args:
            level: Logging level
            msg: Message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        extra = kwargs.get("extra", {})
        extra["pattern_type"] = self.pattern_type
        kwargs["extra"] = extra
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
