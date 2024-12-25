import torch
from pathlib import Path
from huggingface_hub import HfApi, upload_file
from datetime import datetime
import os
from dotenv import load_dotenv

def upload_to_hf(
    repo_id: str = "leonvanbokhorst/topic-drift-detector",
    token: str = None,
    model_path: str = "models/best_topic_drift_model.pt",
    metrics_path: str = "models/full_training_metrics.txt",
    curves_path: str = "models/full_training_curves.png"
):
    """Upload model, metrics, and training curves to Hugging Face.
    
    Args:
        repo_id: Hugging Face repository ID
        token: Hugging Face API token (if None, will try to load from HF_TOKEN env var)
        model_path: Path to saved model
        metrics_path: Path to metrics file
        curves_path: Path to training curves plot
    """
    print("\n=== Uploading to Hugging Face ===")
    
    # Load token from environment if not provided
    if token is None:
        load_dotenv()
        token = os.getenv('HF_TOKEN')
        if token is None:
            raise ValueError("No token provided and HF_TOKEN not found in environment")
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Check if files exist
    model_file = Path(model_path)
    metrics_file = Path(metrics_path)
    curves_file = Path(curves_path)
    
    if not all([model_file.exists(), metrics_file.exists(), curves_file.exists()]):
        raise FileNotFoundError("One or more required files not found")
    
    # Create version tag
    version_tag = datetime.now().strftime("v%Y%m%d_%H%M%S")
    
    try:
        # Upload model
        print(f"Uploading model to {repo_id}")
        model_url = api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo=f"models/{version_tag}/topic_drift_model.pt",
            repo_id=repo_id,
            token=token
        )
        print(f"Model uploaded: {model_url}")
        
        # Upload metrics
        print("Uploading metrics")
        metrics_url = api.upload_file(
            path_or_fileobj=str(metrics_file),
            path_in_repo=f"metrics/{version_tag}/training_metrics.txt",
            repo_id=repo_id,
            token=token
        )
        print(f"Metrics uploaded: {metrics_url}")
        
        # Upload training curves
        print("Uploading training curves")
        curves_url = api.upload_file(
            path_or_fileobj=str(curves_file),
            path_in_repo=f"plots/{version_tag}/training_curves.png",
            repo_id=repo_id,
            token=token
        )
        print(f"Training curves uploaded: {curves_url}")
        
        # Create model card content
        model_card = f"""---
language: en
tags:
- topic-drift
- conversation-analysis
- pytorch
- attention
- lstm
license: mit
datasets:
- leonvanbokhorst/topic-drift-v2
metrics:
- rmse
- r2_score
model-index:
- name: topic-drift-detector
  results:
  - task: 
      type: topic-drift-detection
      name: Topic Drift Detection
    dataset:
      name: leonvanbokhorst/topic-drift-v2
      type: conversations
    metrics:
      - name: Test RMSE
        type: rmse
        value: 0.0144
      - name: Test R²
        type: r2
        value: 0.8666
      - name: Test Loss 
        type: loss
        value: 0.0002
---

# Topic Drift Detector Model

## Version: {version_tag}

This model detects topic drift in conversations using an enhanced hierarchical attention-based architecture. Trained on the [leonvanbokhorst/topic-drift-v2](https://huggingface.co/datasets/leonvanbokhorst/topic-drift-v2) dataset.

## Model Architecture
- Multi-head attention mechanism (4 heads, head dimension 128)
- Hierarchical pattern detection with multi-scale analysis
- Explicit transition point detection with linguistic markers
- Pattern-aware self-attention mechanism
- Dynamic window augmentation
- Contrastive learning with pattern-aware sampling
- Adversarial training with pattern-aware perturbations

### Key Components:
1. **Embedding Processor**:
   - Input dimension: 1024
   - Hidden dimension: 512
   - Dropout rate: 0.35
   - PreNorm layers with residual connections

2. **Attention Blocks**:
   - 3 layers of attention
   - 4 attention heads
   - Feed-forward dimension: 2048
   - Learned position encodings

3. **Pattern Detection**:
   - Hierarchical LSTM layers
   - Bidirectional processing
   - Multi-scale pattern analysis
   - Pattern classification with 7 types

4. **Transition Detection**:
   - Linguistic marker attention
   - Explicit transition scoring
   - Marker-based context integration

## Performance Metrics
```txt
=== Full Training Results ===
Best Validation RMSE: 0.0142
Best Validation R²: 0.8711

=== Test Set Results ===
Loss: 0.0002
RMSE: 0.0144
R²: 0.8666
```

## Training Details
- Dataset: 6400 conversations (5120 train, 640 val, 640 test)
- Window size: 8 turns
- Batch size: 32
- Learning rate: 0.0001 with cosine decay
- Warmup steps: 100
- Early stopping patience: 15
- Max gradient norm: 1.0
- Mixed precision training (AMP)
- Base embeddings: BAAI/bge-m3

### Training Enhancements:
1. **Dynamic Window Augmentation**:
   - Adaptive window sizes
   - Interpolation-based resizing
   - Maintains temporal consistency

2. **Contrastive Learning**:
   - Pattern-aware positive/negative sampling
   - Temperature-scaled similarities
   - Weighted combination of embeddings

3. **Adversarial Training**:
   - Pattern-aware perturbations
   - Self-distillation loss
   - Epsilon ball projection

## Usage Example
```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load base embedding model
base_model = AutoModel.from_pretrained('BAAI/bge-m3')
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

# Load topic drift detector
model = torch.load('models/{version_tag}/topic_drift_model.pt')
model.eval()

# Prepare conversation window (8 turns)
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
    embeddings = base_model(**inputs).last_hidden_state.mean(dim=1)  # [8, 1024]
    
    # Reshape for model input [1, 8*1024]
    conversation_embeddings = embeddings.view(1, -1)
    
    # Get drift score
    drift_scores = model(conversation_embeddings)
    
print(f"Topic drift score: {{drift_scores.item():.4f}}")
# Higher scores indicate more topic drift
```

## Pattern Types
The model detects 7 distinct pattern types:
1. "maintain" - No significant drift
2. "gentle_wave" - Subtle topic evolution
3. "single_peak" - One clear transition
4. "multi_peak" - Multiple transitions
5. "ascending" - Gradually increasing drift
6. "descending" - Gradually decreasing drift
7. "abrupt" - Sudden topic change

## Limitations
- Works best with English conversations
- Requires exactly 8 turns of conversation
- Each turn should be between 1-512 tokens
- Relies on BAAI/bge-m3 embeddings
- May be sensitive to conversation style variations

## Training Curves
![Training Curves](plots/{version_tag}/training_curves.png)
"""
        
        # Upload model card
        print("Uploading model card")
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token
        )
        
        print(f"\nSuccessfully uploaded all files to {repo_id}")
        print(f"Version tag: {version_tag}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")
        raise

if __name__ == "__main__":
    # Upload to Hugging Face using token from .env
    upload_to_hf() 