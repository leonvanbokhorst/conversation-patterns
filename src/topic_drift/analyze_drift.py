import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from topic_drift.data_loader import load_from_huggingface
from topic_drift.data_prep import prepare_training_data
from topic_drift.nn_topic_drift_poc_v2 import EnhancedTopicDriftDetector
import numpy as np

def analyze_drift_distribution():
    """Analyze the distribution of drift scores from the trained model."""
    print("\n=== Analyzing Drift Score Distribution ===")
    
    # Load data
    data = load_from_huggingface("leonvanbokhorst/topic-drift-v2")
    data = prepare_training_data(data, window_size=8, batch_size=32)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset
    test_embeddings = data.test_embeddings.to(device)
    test_labels = data.test_labels.to(device)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load model
    embedding_dim = test_embeddings.shape[1] // 8
    model = EnhancedTopicDriftDetector(embedding_dim).to(device)
    checkpoint = torch.load("models/best_topic_drift_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Collect predictions and targets
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(batch_y.cpu().numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate statistics
    print("\nPrediction Statistics:")
    print(f"Min score: {predictions.min():.4f}")
    print(f"Max score: {predictions.max():.4f}")
    print(f"Mean score: {predictions.mean():.4f}")
    print(f"Std score: {predictions.std():.4f}")
    
    print("\nTarget Statistics:")
    print(f"Min score: {targets.min():.4f}")
    print(f"Max score: {targets.max():.4f}")
    print(f"Mean score: {targets.mean():.4f}")
    print(f"Std score: {targets.std():.4f}")
    
    # Plot distributions
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.5, label='Predictions', density=True)
    plt.hist(targets, bins=50, alpha=0.5, label='Targets', density=True)
    plt.xlabel('Drift Score')
    plt.ylabel('Density')
    plt.title('Distribution of Drift Scores')
    plt.legend()
    
    # Box plot
    plt.subplot(1, 2, 2)
    box_data = [predictions, targets]
    box_labels = ['Predictions', 'Targets']
    plt.boxplot(box_data, labels=box_labels)
    plt.ylabel('Drift Score')
    plt.title('Drift Score Ranges')
    
    plt.tight_layout()
    plt.savefig('models/drift_distributions_comparison.png')
    plt.close()

if __name__ == "__main__":
    analyze_drift_distribution() 