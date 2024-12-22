# Conversational Patterns Research

[![Test](https://github.com/leonvanbokhorst/conversational-patts/actions/workflows/test.yml/badge.svg)](https://github.com/leonvanbokhorst/conversational-patts/actions/workflows/test.yml)

Implementation of human-like conversational patterns for Virtual Humans, focusing on turn-taking, context awareness, response variation, and repair strategies.

## Project Overview

This research project implements and validates key conversational patterns that contribute to more natural human-like interactions in AI systems. The implementation focuses on:

- Turn-taking mechanisms
- Context awareness
- Response variation
- Repair strategies

## Project Structure

```
conversational_patterns/
├── src/
│   └── conversational_patterns/
│       ├── core/          # Core system components
│       ├── patterns/      # Pattern implementations
│       ├── utils/         # Utility functions
│       └── config/        # Configuration management
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── behavioral/       # Pattern-specific tests
├── docs/
│   ├── api/              # API documentation
│   └── architecture/     # Architecture decisions
└── scripts/              # Utility scripts
```

## Setup

### Prerequisites

- Python 3.9 or higher
- Redis server
- Virtual environment tool

### Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

### Development

1. Run tests:

```bash
pytest
```

2. Run type checking:

```bash
mypy src
```

3. Format code:

```bash
black src tests
isort src tests
```

## License

MIT License
