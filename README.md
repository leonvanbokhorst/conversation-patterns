# Topic Drift Detection in Conversations

A deep learning model for detecting and measuring topic drift in conversations using attention mechanisms and pattern recognition.

## Overview

This project implements an enhanced neural network architecture for detecting topic drift in conversational data. The model combines multi-head attention, bidirectional LSTM, and semantic bridge detection to track how conversations evolve and transition between topics.

## Model Architecture

### Key Components

1. **Multi-Head Attention**
   - 4 parallel attention heads for diverse feature capture
   - Each head processes different aspects of the conversation
   - Local context attention with dimension: hidden_dim/num_heads
   - Global context attention with positional encoding
   - Dynamic weight generation for adaptive attention
   - Residual connections and layer normalization

2. **Pattern Detection**
   - Bidirectional LSTM with 3 layers
   - Hidden dimension: 512
   - Dropout rate: 0.2 for regularization
   - Concatenated forward/backward states
   - Pattern classification into 7 drift types:
     * Maintain (no drift)
     * Gentle wave (subtle evolution)
     * Single peak (clear transition)
     * Multi peak (multiple transitions)
     * Ascending (increasing drift)
     * Descending (decreasing drift)
     * Abrupt (sudden topic change)

3. **Transition Analysis**
   - Dynamic transition scoring using similarity thresholds
   - Semantic bridge detection between turns
   - Four transition types with weighted scoring:
     * Topic maintenance (0.11 weight)
     * Smooth transition (0.15 weight)
     * Topic shift (0.18 weight)
     * Abrupt change (0.21 weight)
   - Similarity thresholds:
     * High: > 0.8
     * Medium: > 0.6
     * Low: > 0.4

### Processing Pipeline

1. **Input Processing**
   ```
   Input: [batch_size, sequence_length * embedding_dim]
   │
   ├─► Reshape: [batch_size, 8, 1024]
   │   
   ├─► Embedding Processing
   │   ├─► Linear(1024 → 512)
   │   ├─► LayerNorm + ReLU
   │   ├─► Dropout(0.2)
   │   ├─► Linear(512 → 512)
   │   ├─► LayerNorm + ReLU
   │   └─► Dropout(0.2)
   │
   └─► Add Positional Encoding
   ```

2. **Attention Mechanism**
   ```
   Processed Embeddings
   │
   ├─► Local Attention (per head)
   │   ├─► Linear(512 → 128)
   │   ├─► ReLU
   │   ├─► Linear(128 → 1)
   │   └─► Softmax
   │
   ├─► Global Attention
   │   ├─► Linear(512 → 512)
   │   ├─► Tanh
   │   ├─► Linear(512 → 4)
   │   └─► Softmax
   │
   └─► Pattern Detection
       ├─► BiLSTM(512 → 512 * 2)
       └─► Pattern Classification
   ```

3. **Final Scoring**
   ```
   Combined Features
   │
   ├─► Context Vector
   │   └─► Attention-weighted sum
   │
   ├─► Base Score
   │   └─► Regression(512 → 1)
   │
   ├─► Pattern Factor
   │   └─► Pattern weights [0.8-1.3]
   │
   ├─► Transition Factor
   │   └─► 1 + mean(transitions)
   │
   └─► Final Score
       └─► Clamp(0, 1)
   ```

### Implementation Details

- **Embedding Processor**: Dual-layer network with residual connections
- **Position Encoding**: Learned parameter (8 × hidden_dim)
- **LSTM Configuration**: 
  * 3 layers, bidirectional
  * Hidden size: 512
  * Dropout: 0.2
  * Input size matches hidden dimension
- **Pattern Classifier**:
  * Input: 2 * hidden_dim (concatenated bidirectional)
  * Hidden layer with ReLU activation
  * Output: 7 pattern probabilities
- **Weight Generator**:
  * Input: 2 * hidden_dim
  * Output: 3 attention weights
  * Softmax normalized

## Performance

Current model achieves:
- Test RMSE: 0.0129
- Test R²: 0.8373
- Validation RMSE: 0.0107
- Validation R²: 0.8867

## Installation

[![Test](https://github.com/leonvanbokhorst/conversational-patts/actions/workflows/test.yml/badge.svg)](https://github.com/leonvanbokhorst/conversational-patts/actions/workflows/test.yml)

Implementation of human-like conversational patterns for Virtual Humans, focusing on turn-taking, context awareness, response variation, and repair strategies.

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
import torch
from topic_drift.nn_topic_drift_poc import EnhancedTopicDriftDetector

# Load model
model = torch.load('models/best_topic_drift_model.pt')
model.eval()

# Prepare input
# Shape: [batch_size, sequence_length * embedding_dim]
# where sequence_length = 8 (conversation window)
# and embedding_dim depends on your embedding model

# Get predictions
with torch.no_grad():
    drift_scores = model(conversation_embeddings)
    
# drift_scores will be between 0 and 1
# where higher values indicate more topic drift
```

## Training

The model was trained on conversation data with:
- Window size: 8 turns
- Batch size: 32
- Learning rate: 0.0001
- Early stopping patience: 10
- Total epochs: 37 (early stopped)

## Model Weights

Pre-trained model weights are available on Hugging Face:
- Repository: [lonnstyle/topic-drift-detector](https://huggingface.co/lonnstyle/topic-drift-detector)
- Latest version: Check repository for most recent release

## Future Work

1. Multilingual Support
   - Integration of multilingual conversation datasets
   - Cross-lingual topic drift patterns
   - Culture-specific transition markers

2. Enhanced Pattern Recognition
   - More sophisticated drift patterns
   - Domain-specific adaptations
   - Temporal pattern analysis

3. Real-time Processing
   - Streaming conversation analysis
   - Online learning capabilities
   - Performance optimization

## License

MIT License - See LICENSE file for details

## Citation

If you use this model in your research, please cite:

```bibtex
@software{topic_drift_detector,
  title = {Topic Drift Detection in Conversations},
  author = {Leon van Bokhorst},
  year = {2024},
  url = {https://github.com/yourusername/topic-drift-detection}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
