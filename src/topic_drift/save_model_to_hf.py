import os
import torch
from huggingface_hub import HfApi
from datetime import datetime
from topic_drift.nn_topic_drift_poc_v2 import EnhancedTopicDriftDetector
from pathlib import Path
from dotenv import load_dotenv

def save_model_to_hf(token: str = None):
    """Upload model, metrics, and documentation to Hugging Face."""
    print("\n=== Uploading to Hugging Face ===")
    
    # Load token from environment if not provided
    if token is None:
        load_dotenv()
        token = os.getenv('HF_TOKEN')
        if token is None:
            raise ValueError("No token provided and HF_TOKEN not found in environment")
    
    # Initialize the model
    model = EnhancedTopicDriftDetector(input_dim=1024, hidden_dim=512)
    
    # Load the trained weights
    model_path = Path('models/topic_drift_model.pt')
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create a timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_tag = f"v{timestamp}"
    model_dir = Path(f"models/{version_tag}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    save_path = model_dir / "topic_drift_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_dim': 512,
        'input_dim': 1024,
        'hyperparameters': {
            'hidden_dim': 512,
            'input_dim': 1024,
            'num_heads': 4,
            'dropout': 0.1
        },
        'metrics': {
            'rmse': checkpoint.get('val_rmse', 0.0),
            'r2': checkpoint.get('train_rmse', 0.0)
        }
    }, save_path)
    
    # Upload to Hugging Face
    api = HfApi(token=token)
    
    # Upload model
    print("Uploading model...")
    model_url = api.upload_file(
        path_or_fileobj=str(save_path),
        path_in_repo=f"models/{version_tag}/topic_drift_model.pt",
        repo_id="leonvanbokhorst/topic-drift-detector",
        repo_type="model"
    )
    print(f"Model uploaded to: {model_url}")
    
    # Create model card content
    model_card = f"""---
language: en
tags:
- topic-drift
- conversation-analysis
- pytorch
- attention
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
        value: {checkpoint.get('val_rmse', 0.0):.4f}
      - name: Test R²
        type: r2
        value: {checkpoint.get('train_rmse', 0.0):.4f}
---

# Topic Drift Detector Model

## Version: {version_tag}

This model detects topic drift in conversations using an efficient attention-based architecture. Trained on the [leonvanbokhorst/topic-drift-v2](https://huggingface.co/datasets/leonvanbokhorst/topic-drift-v2) dataset.

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

## Performance Metrics
```txt
=== Test Set Results ===
RMSE: {checkpoint.get('val_rmse', 0.0):.4f}
R��: {checkpoint.get('train_rmse', 0.0):.4f}
```

## Training Details
- Dataset: 6400 conversations (5120 train, 640 val, 640 test)
- Window size: 8 turns
- Batch size: 32
- Learning rate: 0.0001
- Early stopping patience: 15
- Distribution regularization weight: 0.1
- Target standard deviation: 0.2
- Base embeddings: BAAI/bge-m3

## Usage Example

```python
# Install dependencies
pip install torch transformers huggingface_hub

# Import required packages
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Load base model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model = AutoModel.from_pretrained('BAAI/bge-m3').to(device)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')

# Download and load topic drift model
model_path = hf_hub_download(
    repo_id='leonvanbokhorst/topic-drift-detector',
    filename='models/{version_tag}/topic_drift_model.pt'
)
checkpoint = torch.load(model_path, weights_only=True, map_location=device)
model = EnhancedTopicDriftDetector(
    input_dim=1024,
    hidden_dim=checkpoint['hyperparameters']['hidden_dim']
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
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

# Process conversation
with torch.no_grad():
    # Get embeddings
    inputs = tokenizer(conversation, padding=True, truncation=True, return_tensors='pt')
    inputs = dict((k, v.to(device)) for k, v in inputs.items())
    embeddings = base_model(**inputs).last_hidden_state.mean(dim=1)
    
    # Get drift score
    conversation_embeddings = embeddings.view(1, -1)
    drift_score = model(conversation_embeddings)
    print(f"Topic drift score: {{drift_score.item():.4f}}")
```

## Limitations
- Works best with English conversations
- Requires exactly 8 turns of conversation
- Each turn should be between 1-512 tokens
- Relies on BAAI/bge-m3 embeddings
"""
    
    # Upload model card
    print("Uploading model card...")
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id="leonvanbokhorst/topic-drift-detector",
        repo_type="model"
    )
    
    print(f"\nSuccessfully uploaded model and documentation")
    print(f"Version tag: {version_tag}")

if __name__ == "__main__":
    save_model_to_hf() 