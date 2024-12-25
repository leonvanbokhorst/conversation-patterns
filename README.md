# Topic Drift Detection in Conversation

[![Test](https://github.com/leonvanbokhorst/conversational-patts/actions/workflows/test.yml/badge.svg)](https://github.com/leonvanbokhorst/conversational-patts/actions/workflows/test.yml)

A deep learning model for detecting and measuring topic drift in conversations using hierarchical attention mechanisms and advanced pattern recognition.

## Overview

This project implements a neural network architecture for detecting topic drift in conversational data. The model combines hierarchical pattern detection, multi-head attention, explicit transition detection, and pattern-aware self-attention to track how conversations evolve and transition between topics.

## Model Architecture

### Key Components

1. **Multi-Head Attention**
   - 4 parallel attention heads (head dimension: 128)
   - Local and global context processing
   - PreNorm layers with residual connections
   - Learned positional encodings
   - Feed-forward dimension: 2048
   - Dropout rate: 0.35 for strong regularization

2. **Hierarchical Pattern Detection**
   - Multi-scale pattern analysis
   - Bidirectional LSTM layers
   - Downsampling and projection layers
   - Pattern classification into 7 drift types:
     * Maintain (no drift)
     * Gentle wave (subtle evolution)
     * Single peak (clear transition)
     * Multi peak (multiple transitions)
     * Ascending (increasing drift)
     * Descending (decreasing drift)
     * Abrupt (sudden topic change)

3. **Transition Detection**
   - Explicit transition point detection
   - Linguistic marker attention
   - Marker-based context integration
   - Dynamic transition scoring
   - Semantic bridge detection between turns

4. **Pattern Self-Attention**
   - Pattern-aware attention mechanism
   - Dynamic weight generation
   - Gating mechanism for pattern importance
   - Pattern interaction modeling

### Training Enhancements

1. **Dynamic Window Augmentation**
   - Adaptive window sizes
   - Interpolation-based resizing
   - Maintains temporal consistency
   - Replication padding for shorter windows

2. **Contrastive Learning**
   - Pattern-aware positive/negative sampling
   - Temperature-scaled similarities (0.07)
   - Weighted combination of similarities:
     * Embedding similarity: 0.4
     * Pattern similarity: 0.4
     * Transition similarity: 0.2

3. **Adversarial Training**
   - Pattern-aware perturbations
   - Self-distillation loss
   - Epsilon ball projection
   - Multi-step adversarial updates

### Processing Pipeline

1. **Input Processing**
   ```
   Input: [batch_size, sequence_length * embedding_dim]
   │
   ├─► Reshape: [batch_size, 8, 1024]
   │   
   ├─► Embedding Processing
   │   ├─► PreNorm + Linear(1024 → 512)
   │   ├─► GELU
   │   ├─► Dropout(0.35)
   │   ├─► Linear(512 → 512)
   │   └─► Dropout(0.35)
   │
   └─► Add Learned Position Encoding
   ```

2. **Pattern Recognition**
   ```
   Processed Embeddings
   │
   ├─► Attention Blocks (×3)
   │   ├─► Multi-Head Attention
   │   │   ├─► PreNorm
   │   │   ├─► 4 heads × 128 dim
   │   │   └─► Residual connection
   │   │
   │   └─► Feed Forward
   │       ├─► PreNorm
   │       ├─► Linear(512 → 2048)
   │       ├─► GELU + Dropout
   │       ├─► Linear(2048 → 512)
   │       └─► Residual connection
   │
   ├─► Hierarchical Pattern Detection
   │   ├─► Multi-scale LSTM layers
   │   ├─► Downsampling convolutions
   │   └─► Pattern features
   │
   └─► Transition Detection
       ├─► Marker attention
       ├─► Context integration
       └─► Transition scores
   ```

3. **Final Prediction**
   ```
   Combined Features
   │
   ├─► Pattern Features
   │   ├─► Dimension adapter
   │   ├─► Pattern classification
   │   └─► Pattern interaction
   │
   ├─► Transition Features
   │   └─► Mean transition scores
   │
   └─► Final Regression
       ├─► PreNorm + Linear
       ├─► GELU + Dropout
       └─► Sigmoid activation
   ```

## Performance

Current model achieves:
- Test RMSE: 0.0144
- Test R²: 0.8666
- Best Validation RMSE: 0.0142
- Best Validation R²: 0.8711

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

```python
import torch
from transformers import AutoModel, AutoTokenizer
from topic_drift.nn_topic_drift_poc_v2 import EnhancedTopicDriftDetector

def load_model(model_path: str = 'models/best_topic_drift_model.pt') -> EnhancedTopicDriftDetector:
    """Load the topic drift model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=True)
    
    # Create model with same hyperparameters
    model = EnhancedTopicDriftDetector(
        input_dim=1024,  # BGE-M3 embedding dimension
        hidden_dim=checkpoint['hyperparameters']['hidden_dim']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Load base embedding model
base_model = AutoModel.from_pretrained('BAAI/bge-m3')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

# Load topic drift detector
model = load_model()
model.eval()

# Example conversation with topic drift
conversation = [
    "How was your weekend?",
    "It was great! Went hiking.",
    "Which trail did you take?",
    "The mountain loop trail.",
    "That's nice. By the way, did you watch the game?",  # Topic shift
    "Yes! What an amazing match!",
    "The final score was incredible.",
    "I couldn't believe that last-minute goal."
]

# Get embeddings
with torch.no_grad():
    inputs = tokenizer(conversation, padding=True, truncation=True, return_tensors='pt')
    embeddings = base_model(**inputs).last_hidden_state.mean(dim=1)  # [8, 1024]
    
    # Reshape for model input [1, 8*1024]
    conversation_embeddings = embeddings.view(1, -1)
    
    # Get drift score
    drift_scores = model(conversation_embeddings)
    
print("\nExample Conversation Analysis")
print("-" * 50)
print("Conversation turns:")
for i, turn in enumerate(conversation, 1):
    print(f"{i}. {turn}")
print("-" * 50)
print(f"Topic drift score: {drift_scores.item():.4f}")
print("Note: Higher scores indicate more topic drift")

# Example output:
# Topic drift score: 0.4454 - Moderate to high drift
# Shows clear transition from hiking to sports discussion
```

## Training Details

The model was trained on conversation data with:
- Dataset: 6400 conversations (5120 train, 640 val, 640 test)
- Repository: [leonvanbokhorst/topic-drift-v2](https://huggingface.co/leonvanbokhorst/topic-drift-v2)
- Window size: 8 turns
- Batch size: 32
- Learning rate: 0.0001 with cosine decay
- Warmup steps: 100
- Early stopping patience: 15
- Max gradient norm: 1.0
- Mixed precision training (AMP)
- Base embeddings: BAAI/bge-m3

## Model Weights

Pre-trained model weights are available on Hugging Face:
- Repository: [leonvanbokhorst/topic-drift-detector](https://huggingface.co/leonvanbokhorst/topic-drift-detector)
- Latest version: Check repository for most recent release

## Future Work

1. Multilingual Support
   - Integration of multilingual conversation datasets
   - Cross-lingual topic drift patterns
   - Culture-specific transition markers

2. Enhanced Pattern Recognition
   - Temporal pattern analysis
   - Domain-specific adaptations
   - Few-shot pattern learning

3. Real-time Processing
   - Streaming conversation analysis
   - Incremental pattern updates
   - Optimized inference pipeline

## License

MIT License - See LICENSE file for details

## Citation

If you use this model in your research, please cite:

```bibtex
@software{topic_drift_detector,
  title = {Topic Drift Detection in Conversations},
  author = {Leon van Bokhorst},
  year = {2024},
  url = {https://github.com/leonvanbokhorst/topic-drift-detection}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
