import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict
from topic_drift.data_loader import load_from_huggingface
from topic_drift.data_prep import prepare_training_data, DataSplit
from topic_drift.llm_wrapper import OllamaWrapper
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


class TopicDriftDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """Initialize the topic drift detection model.

        Args:
            input_dim: Dimension of a single turn embedding
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.input_dim = input_dim

        # Embedding processing layers
        self.embedding_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        # Final regression layers
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Ensure output is between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass processing a window of embeddings.

        Args:
            x: Tensor of shape (batch_size, window_size * embedding_dim)

        Returns:
            Tensor of shape (batch_size, 1) with drift scores between 0 and 1
        """
        batch_size = x.shape[0]
        window_size = x.shape[1] // self.input_dim

        # Reshape to (batch_size, window_size, embedding_dim)
        x = x.view(batch_size, window_size, self.input_dim)

        # Process each embedding
        processed = self.embedding_processor(
            x
        )  # Shape: (batch_size, window_size, hidden_dim)

        # Apply attention
        attention_weights = self.attention(
            processed
        )  # Shape: (batch_size, window_size, 1)
        context = torch.sum(
            attention_weights * processed, dim=1
        )  # Shape: (batch_size, hidden_dim)

        # Final regression
        return self.regressor(context)


def evaluate_model(
    model: TopicDriftDetector,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            total_loss += loss.item()

            all_preds.extend(outputs.squeeze().tolist())
            all_targets.extend(batch_y.tolist())

    return {
        "loss": total_loss / len(dataloader),
        "mse": mean_squared_error(all_targets, all_preds),
        "rmse": mean_squared_error(all_targets, all_preds, squared=False),
        "r2": r2_score(all_targets, all_preds),
    }


def train_model(
    data: DataSplit,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 3,
) -> Tuple[TopicDriftDetector, Dict[str, list]]:
    """Train the topic drift detection model.

    Args:
        data: DataSplit object containing all data splits
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        early_stopping_patience: Number of epochs to wait for improvement

    Returns:
        Tuple[TopicDriftDetector, dict]: Trained model and training metrics dictionary
    """
    # Create data loaders
    train_dataset = TensorDataset(data.train_embeddings, data.train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(data.val_embeddings, data.val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and training components
    embedding_dim = data.train_embeddings.shape[1] // 3  # Assuming window_size=3
    model = TopicDriftDetector(embedding_dim)
    criterion = nn.MSELoss()  # Use MSE loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize metrics tracking
    metrics = {
        "train_losses": [],
        "train_rmse": [],
        "train_r2": [],
        "val_losses": [],
        "val_rmse": [],
        "val_r2": [],
    }

    # Early stopping setup
    best_val_rmse = float("inf")
    patience_counter = 0
    best_model_state = None

    # Training loop with tqdm
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_metrics = {"loss": 0.0, "all_preds": [], "all_targets": []}

        batch_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")
        for batch_x, batch_y in batch_pbar:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect predictions and metrics
            train_metrics["all_preds"].extend(outputs.squeeze().detach().tolist())
            train_metrics["all_targets"].extend(batch_y.tolist())
            train_metrics["loss"] += loss.item()
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # Calculate training metrics
        train_results = {
            "loss": train_metrics["loss"] / len(train_loader),
            "rmse": mean_squared_error(
                train_metrics["all_targets"], train_metrics["all_preds"], squared=False
            ),
            "r2": r2_score(train_metrics["all_targets"], train_metrics["all_preds"]),
        }

        # Validation phase
        val_results = evaluate_model(model, val_loader, criterion)

        # Update metrics
        metrics["train_losses"].append(train_results["loss"])
        metrics["train_rmse"].append(train_results["rmse"])
        metrics["train_r2"].append(train_results["r2"])
        metrics["val_losses"].append(val_results["loss"])
        metrics["val_rmse"].append(val_results["rmse"])
        metrics["val_r2"].append(val_results["r2"])

        # Early stopping check
        if val_results["rmse"] < best_val_rmse:
            best_val_rmse = val_results["rmse"]
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Update progress bar
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_results['loss']:.4f}",
                "train_rmse": f"{train_results['rmse']:.4f}",
                "val_loss": f"{val_results['loss']:.4f}",
                "val_rmse": f"{val_results['rmse']:.4f}",
            }
        )

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, metrics


def main():
    """Load data, prepare it, and train the model."""
    # Load conversation data from Hugging Face
    conversation_data = load_from_huggingface()

    # Prepare training data with splits
    data = prepare_training_data(
        conversation_data,
        window_size=8,  # Use 8 turns for context
        batch_size=32,  # Reduced batch size for larger windows
        max_workers=8,
        force_recompute=True,  # Force recompute since we changed the data format
    )

    # Train model with enhanced metrics
    model, metrics = train_model(
        data,
        batch_size=16,  # Reduced batch size for training due to larger windows
        epochs=20,  # Increased epochs for better convergence
        learning_rate=0.0005,  # Reduced learning rate for stability
        early_stopping_patience=5,  # Increased patience for better convergence
    )

    # Evaluate on test set
    test_dataset = TensorDataset(data.test_embeddings, data.test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)  # Reduced batch size
    test_results = evaluate_model(model, test_loader, nn.MSELoss())

    # Print final metrics
    print("\nTraining Results:")
    print(f"Best Validation RMSE: {min(metrics['val_rmse']):.4f}")
    print(f"Best Validation R²: {max(metrics['val_r2']):.4f}")
    print("\nTest Set Results:")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"RMSE: {test_results['rmse']:.4f}")
    print(f"R²: {test_results['r2']:.4f}")

    # Example predictions
    model.eval()
    with torch.no_grad():
        sample_inputs = data.test_embeddings[:5]
        sample_targets = data.test_labels[:5]
        predictions = model(sample_inputs).squeeze()

        print("\nSample Predictions vs Actual:")
        for pred, target in zip(predictions, sample_targets):
            print(
                f"Predicted drift: {pred.item():.4f}, Actual drift: {target.item():.4f}"
            )


if __name__ == "__main__":
    main()
