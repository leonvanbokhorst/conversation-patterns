# Topic Drift Detection in Conversation

[![Test](https://github.com/leonvanbokhorst/conversational-patts/actions/workflows/test.yml/badge.svg)](https://github.com/leonvanbokhorst/conversational-patts/actions/workflows/test.yml)

A PyTorch-based model for detecting topic drift in conversations using an efficient attention-based architecture.

## Overview

This project implements a neural network model that detects topic drift in conversations. It uses BAAI/bge-m3 embeddings and a streamlined attention mechanism to analyze conversation flow and identify when topics change.

## Model Architecture

### Key Components:
1. **Input Processing**:
   - Input dimension: 1024 (BGE-M3 embeddings)
   - Hidden dimension: 512
   - Sequence length: 8 turns

2. **Attention Block**:
   - Multi-head attention (4 heads)
   - PreNorm layers with residual connections
   - Dropout rate: 0.1

3. **Feed-Forward Network**:
   - Two-layer MLP with GELU activation
   - Hidden dimension: 512 -> 2048 -> 512
   - Residual connections

4. **Output Layer**:
   - Two-layer MLP: 512 -> 256 -> 1
   - GELU activation
   - Direct sigmoid output for [0,1] range

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

```python
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

def load_model(repo_id: str = "leonvanbokhorst/topic-drift-detector"):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Download latest model weights
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="models/v20241226_112605/topic_drift_model.pt", #latest
        force_download=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    
    # Create model
    model = EnhancedTopicDriftDetector(
        input_dim=1024,
        hidden_dim=checkpoint['hyperparameters']['hidden_dim']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, device

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = AutoModel.from_pretrained('BAAI/bge-m3').to(device)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model, _ = load_model()
model.eval()

# Example conversation
conversation = [
    "How was your weekend?",
    "It was great! Went hiking.",
    "Which trail did you take?",
    "The mountain loop trail.",
    "That's nice. By the way, did you watch the game?",
    "Yes! What an amazing match!",
    "The final score was incredible.",
    "I couldn't believe that last-minute goal."
]

# Get embeddings
with torch.no_grad():
    inputs = tokenizer(conversation, padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = base_model(**inputs).last_hidden_state.mean(dim=1)  # [8, 1024]
    conversation_embeddings = embeddings.view(1, -1)
    drift_score = model(conversation_embeddings)

print(f"Topic drift score: {drift_score.item():.4f}")
# Higher scores indicate more topic drift
```

## Training Details

- Dataset: 6400 conversations (5120 train, 640 val, 640 test)
- Repository: [leonvanbokhorst/topic-drift-v2](https://huggingface.co/leonvanbokhorst/topic-drift-v2)
- Window size: 8 turns
- Batch size: 32
- Learning rate: 0.0001
- Early stopping patience: 15
- Distribution regularization weight: 0.1
- Target standard deviation: 0.2
- Base embeddings: BAAI/bge-m3

## Limitations

- Works best with English conversations
- Requires exactly 8 turns of conversation
- Each turn should be between 1-512 tokens
- Relies on BAAI/bge-m3 embeddings

## License

MIT License
