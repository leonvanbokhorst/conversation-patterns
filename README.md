# 🧠 Conversational Patterns Lab

> Where we teach AIs to be less awkward at parties!

## 🔬 Experiments Overview

This repo contains several experiments exploring different aspects of AI conversation:

### 1. 🎭 Narrative Mind

- Virtual humans with evolving mental models
- Memory systems that actually remember stuff (unlike that one time I forgot where I put my keys... while holding them)
- Located in: `src/narrative_mind/`

### 2. 🎯 Topic Drift

- Neural networks that try to stay on topic (better than most humans at meetings)
- Experiments with conversation flow and coherence
- Located in: `src/topic_drift/`

### 3. 🎨 Prompt Crafting

- Making LLMs respond more naturally
- Integration patterns for better conversations
- Located in: `src/prompt_crafting/`

### 4. 🎲 Response Variability

- Making AI responses less repetitive and more human-like
- Because nobody likes talking to a broken record
- Located in: `src/response-variability/`

### 5. 🤝 Conversational Patterns

- Core patterns for natural dialogue
- Located in: `src/conversational_patterns/`

## 🚀 Quick Start

```bash
# Clone this repo (you probably already did that, high five! ✋)
git clone https://github.com/yourusername/conversational-patts

# Install dependencies
pip install -r requirements.txt

# Run experiments
# (Each directory has its own README with specific instructions)
```

## 📚 Project Structure

```
src/
├── narrative_mind/      # Virtual humans with memory
├── topic_drift/         # Conversation flow experiments
├── prompt_crafting/     # Natural language generation
├── response-variability/# Response diversity studies
└── conversational_patterns/ # Core dialogue patterns
```

## 🎯 Key Features

- Event-driven processing
- Distributed computation support
- Async LLM integration
- Real-time conversation analysis
- Memory persistence
- Natural language understanding

## 🤓 Technical Stack

- Python (because we're civilized here)
- PyTorch (for the neural magic)
- Transformers (the AI kind, not the robots in disguise)
- FastAPI (for speedy services)
- Redis (because we remember things)

## 📝 Code Standards

We keep it clean and tidy:

- Type hints (because guessing is for fortune tellers)
- Documentation (future you will thank past you)
- Tests (lots of them)
- PEP 8 (because chaos is so 2010)

## 🤝 Contributing

Got ideas? Found a bug? Want to make AI conversations even better?

1. Fork it
2. Branch it
3. Code it
4. Test it
5. PR it

## 📜 License

MIT - Because sharing is caring!

# Enhanced Virtual Human Memory System

A sophisticated memory system that models human-like memory behaviors with emotional differentiation and narrative coherence. This system is designed to create more authentic and emotionally engaging interactions in virtual human simulations.

## Features

### 1. Emotional Differentiation

- Rich emotional state modeling with primary and secondary emotions
- Dynamic emotional intensity tracking
- Emotional congruence in memory clusters
- Emotion-based memory activation

### 2. Thematic Coherence

- Automatic narrative clustering
- Dynamic connection strength adjustment
- Thematic resonance between memories
- Narrative gravity fields

### 3. Memory Dynamics

- Short-term to long-term memory consolidation
- Novelty detection and attention boosting
- Natural memory decay
- Reinforcement through recall

### 4. Visualization

- Memory network visualization
- Narrative cluster visualization
- Emotional state visualization
- Interactive graph exploration

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from memory.enhanced_memory_system import EnhancedMemorySystem

# Create a memory system
memory_system = EnhancedMemorySystem()

# Process a new experience
memory_system.process_new_experience(
    content="Had a great team meeting",
    emotions={
        "joy": 0.8,
        "trust": 0.7,
        "anticipation": 0.6
    },
    context={
        "location": "Office",
        "activity": "meeting",
        "tags": ["work", "collaboration", "communication"]
    }
)
```

### Visualization

```python
from memory.memory_visualizer import MemoryVisualizer

visualizer = MemoryVisualizer()

# Visualize memory network
visualizer.visualize_memory_network(
    memory_system.long_term_storage,
    save_path="memory_network.png"
)

# Visualize narrative clusters
visualizer.visualize_narrative_clusters(
    memory_system.narrative_clusters,
    save_path="narrative_clusters.png"
)
```

## Demo

Run the demonstration script to see the system in action:

```bash
python src/memory/enhanced_memory_demo.py
```

The demo showcases:

1. Emotional differentiation
2. Thematic coherence
3. Memory resonance
4. Visualization capabilities

## System Architecture

### Components

1. **EmotionalState**

   - Manages primary and secondary emotions
   - Tracks emotional intensity and duration
   - Provides emotional energy calculations

2. **MemoryNode**

   - Stores content and context
   - Maintains emotional state
   - Tracks connections and activation

3. **NarrativeCluster**

   - Groups related memories
   - Maintains thematic coherence
   - Calculates cluster gravity

4. **MemoryVisualizer**
   - Creates network visualizations
   - Visualizes narrative clusters
   - Provides emotional state visualization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
