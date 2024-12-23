import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from data_generator import generate_synthetic_data, OllamaWrapper
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
) -> Tuple[TopicDriftDetector, list]:
    """Train the topic drift detection model."""
    # Create data loader
    dataset = TensorDataset(embeddings1, embeddings2, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and training components
    input_dim = embeddings1.shape[1]
    model = TopicDriftDetector(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x1, batch_x2, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_x1, batch_x2)
            loss = criterion(outputs, batch_y.unsqueeze(1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, losses


def main():
    # Generate or load conversation data
    conversation_data = generate_synthetic_data()

    # Create embeddings and labels from the conversation data
    ollama = OllamaWrapper(embedding_model="bge-m3")
    embeddings1, embeddings2, labels = [], [], []

    for conv in conversation_data.conversations:
        turns = conv["turns"]
        # Process consecutive turns
        for i in range(len(turns) - 1):
            turn1, turn2 = turns[i], turns[i + 1]
            emb1 = ollama.get_embeddings(turn1)
            emb2 = ollama.get_embeddings(turn2)

            # Calculate drift based on cosine similarity
            sim_score = cosine_similarity([emb1], [emb2])[0][0]
            drift_label = 1 if sim_score < 0.7 else 0

            embeddings1.append(emb1)
            embeddings2.append(emb2)
            labels.append(drift_label)

    # Convert to tensors
    embeddings1 = torch.tensor(np.array(embeddings1), dtype=torch.float32)
    embeddings2 = torch.tensor(np.array(embeddings2), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Train model
    model, training_losses = train_model(embeddings1, embeddings2, labels)

    # Example prediction
    with torch.no_grad():
        sample_pred = model(embeddings1[:1], embeddings2[:1])
        print(f"\nSample prediction: {sample_pred.item():.4f}")
        print(f"Actual label: {labels[0].item()}")


if __name__ == "__main__":
    main()
