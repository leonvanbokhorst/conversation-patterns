import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from topic_drift.nn_topic_drift_poc_v2 import EnhancedTopicDriftDetector

def load_model(repo_id: str = "leonvanbokhorst/topic-drift-detector"):
    """Load the topic drift model from Hugging Face."""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Download latest model weights
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="models/v20241226_114030/topic_drift_model.pt",
            force_download=True  # Ensure we get the latest version
        )
        print(f"Downloaded model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, weights_only=True, map_location=device)
        print("Loaded checkpoint successfully")
        
        # Create model with same hyperparameters
        model = EnhancedTopicDriftDetector(
            input_dim=1024,  # BGE-M3 embedding dimension
            hidden_dim=checkpoint['hyperparameters']['hidden_dim']
        ).to(device)
        print(f"Created model with hidden_dim={checkpoint['hyperparameters']['hidden_dim']}")
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model state successfully")
        
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

print("\n=== Loading Models ===")

# Get device first
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load base embedding model
print("\nLoading base embedding model...")
base_model = AutoModel.from_pretrained('BAAI/bge-m3').to(device)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
print("Base model loaded successfully")

# Load topic drift detector from Hugging Face
print("\nLoading topic drift detector...")
model, _ = load_model()  # We already have device
model.eval()
print("Topic drift detector loaded successfully")

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

print("\n=== Analyzing Conversation ===")
print("Processing conversation:")
for i, turn in enumerate(conversation, 1):
    print(f"{i}. {turn}")

# Get embeddings
with torch.no_grad():
    # Tokenize and get embeddings
    inputs = tokenizer(conversation, padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    embeddings = base_model(**inputs).last_hidden_state.mean(dim=1)  # [8, 1024]
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    
    # Reshape for model input [1, 8*1024]
    conversation_embeddings = embeddings.view(1, -1)
    print(f"Reshaped to: {conversation_embeddings.shape}")
    
    # Get drift score
    drift_scores = model(conversation_embeddings)
    print("\nPrediction successful")

print("\n=== Results ===")
print(f"Topic drift score: {drift_scores.item():.4f}")
print("Note: Higher scores indicate more topic drift")
