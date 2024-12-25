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

def main():
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

if __name__ == "__main__":
    main()