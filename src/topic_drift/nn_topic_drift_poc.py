import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from topic_drift.data_loader import load_from_huggingface
from topic_drift.data_prep import prepare_training_data
from topic_drift.llm_wrapper import OllamaWrapper
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


class TopicDriftDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """Initialize the topic drift detection model."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass concatenating two embeddings."""
        combined = torch.cat((x1, x2), dim=1)
        return self.network(combined)


def train_model(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> Tuple[TopicDriftDetector, dict]:
    """Train the topic drift detection model.

    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        labels: Labels tensor
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple[TopicDriftDetector, dict]: Trained model and training metrics dictionary
    """
    # Create data loader
    dataset = TensorDataset(embeddings1, embeddings2, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and training components
    input_dim = embeddings1.shape[1]
    model = TopicDriftDetector(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize metrics tracking
    metrics = {
        "losses": [],
        "accuracies": [],
        "precisions": [],
        "recalls": [],
        "f1_scores": [],
    }

    # Training loop with tqdm
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []

        # Batch progress bar
        batch_pbar = tqdm(dataloader, leave=False, desc=f"Epoch {epoch+1}")
        for batch_x1, batch_x2, batch_y in batch_pbar:
            # Forward pass
            outputs = model(batch_x1, batch_x2)
            loss = criterion(outputs, batch_y.unsqueeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect predictions and true labels
            preds = (outputs > 0.5).float().squeeze()
            # Handle both single-item and batch predictions
            if preds.dim() == 0:
                all_preds.append(preds.item())
                all_labels.append(batch_y.item())
            else:
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.tolist())

            epoch_loss += loss.item()
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # Calculate epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        # Update metrics
        metrics["losses"].append(avg_loss)
        metrics["accuracies"].append(accuracy)
        metrics["precisions"].append(precision)
        metrics["recalls"].append(recall)
        metrics["f1_scores"].append(f1)

        # Update progress bar
        epoch_pbar.set_postfix(
            {"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.4f}", "f1": f"{f1:.4f}"}
        )

    return model, metrics


def main():
    """Load data, prepare it, and train the model."""
    # Load conversation data from Hugging Face
    conversation_data = load_from_huggingface()

    # Prepare training data
    embeddings1, embeddings2, labels = prepare_training_data(conversation_data)

    # Train model with enhanced metrics
    model, metrics = train_model(embeddings1, embeddings2, labels)

    # Print final metrics
    print("\nTraining Results:")
    print(f"Final Loss: {metrics['losses'][-1]:.4f}")
    print(f"Final Accuracy: {metrics['accuracies'][-1]:.4f}")
    print(f"Final F1 Score: {metrics['f1_scores'][-1]:.4f}")
    print(f"Final Precision: {metrics['precisions'][-1]:.4f}")
    print(f"Final Recall: {metrics['recalls'][-1]:.4f}")

    # Example prediction
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        sample_pred = model(embeddings1[:1], embeddings2[:1])
        print(f"\nSample prediction: {sample_pred.item():.4f}")
        print(f"Actual label: {labels[0].item()}")


if __name__ == "__main__":
    main()
