import torch
import numpy as np
from pathlib import Path
from topic_drift.data_loader import load_from_huggingface
from topic_drift.data_prep import prepare_training_data
from topic_drift.nn_topic_drift_poc_v2 import EnhancedTopicDriftDetector
from typing import List, Tuple

def get_conversation_text(data, idx: int) -> List[str]:
    """Get the conversation turns for a given index."""
    return data.conversations[idx]['turns']

def analyze_examples():
    """Analyze model predictions on specific examples."""
    print("\n=== Analyzing Example Predictions ===")
    
    # Load data
    data = load_from_huggingface("leonvanbokhorst/topic-drift-v2")
    processed_data = prepare_training_data(data, window_size=8, batch_size=32)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    embedding_dim = processed_data.test_embeddings.shape[1] // 8
    model = EnhancedTopicDriftDetector(embedding_dim).to(device)
    checkpoint = torch.load("models/best_topic_drift_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions for test set
    test_embeddings = processed_data.test_embeddings.to(device)
    test_labels = processed_data.test_labels.to(device)
    
    with torch.no_grad():
        predictions = model(test_embeddings).cpu().numpy().flatten()
        targets = test_labels.cpu().numpy()
    
    print("\nOverall Statistics:")
    print(f"Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
    print(f"Target range: {targets.min():.4f} - {targets.max():.4f}")
    print(f"Mean prediction: {predictions.mean():.4f} (std: {predictions.std():.4f})")
    print(f"Mean target: {targets.mean():.4f} (std: {targets.std():.4f})")
    
    # Find interesting examples
    high_drift = np.argsort(predictions)[-5:]  # Top 5 highest drift
    low_drift = np.argsort(predictions)[:5]    # Top 5 lowest drift
    mid_drift = np.argsort(np.abs(predictions - 0.5))[:5]  # 5 closest to 0.5
    
    # Analyze examples
    def print_example(idx: int, category: str):
        # Get conversation from test set
        test_idx = idx % len(data.conversations)  # Ensure index is in range
        turns = get_conversation_text(data, test_idx)
        pred = predictions[idx]
        target = targets[idx]
        
        print(f"\n=== {category} (Test Index: {idx}, Conv Index: {test_idx}) ===")
        print(f"Predicted drift: {pred:.4f}")
        print(f"Actual drift: {target:.4f}")
        print(f"Error: {abs(pred - target):.4f}")
        print("\nConversation:")
        for i, turn in enumerate(turns, 1):
            print(f"Turn {i}: {turn}")
        print("-" * 80)
    
    print("\n=== High Drift Examples ===")
    for idx in high_drift:
        print_example(idx, "High Drift")
    
    print("\n=== Low Drift Examples ===")
    for idx in low_drift:
        print_example(idx, "Low Drift")
    
    print("\n=== Medium Drift Examples ===")
    for idx in mid_drift:
        print_example(idx, "Medium Drift")

if __name__ == "__main__":
    analyze_examples() 