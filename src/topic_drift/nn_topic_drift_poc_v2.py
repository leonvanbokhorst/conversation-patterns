import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional
from topic_drift.data_loader import load_from_huggingface
from topic_drift.data_prep import prepare_training_data, DataSplit
from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import math
import os
from transformers import get_cosine_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer warnings
torch.multiprocessing.set_sharing_strategy('file_system')  # Fix multiprocessing issues

def distribution_loss(outputs: torch.Tensor, target_std: float = 0.2) -> torch.Tensor:
    """Loss term to encourage wider distribution.
    
    Args:
        outputs: Model predictions
        target_std: Target standard deviation for the outputs
        
    Returns:
        Loss value encouraging the outputs to have the target standard deviation
    """
    batch_std = outputs.std()
    return F.mse_loss(batch_std, torch.tensor(target_std, device=outputs.device))

class EnhancedTopicDriftDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Initialize layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Final regression layer with bias
        self.final_regression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Direct output in [0,1] range
        )
        
    def forward(self, x):
        # Reshape input from [batch_size, seq_len * input_dim] to [batch_size, seq_len, input_dim]
        batch_size = x.shape[0]
        x = x.view(batch_size, 8, self.input_dim)  # 8 is the sequence length
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Prepare input for MultiheadAttention (expects seq_len first)
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Transpose back
        x = attn_output.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        x = self.norm1(x + self.dropout(attn_output.transpose(0, 1)))
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        
        # Apply final regression
        drift_score = self.final_regression(x)  # [batch_size, 1] in range [0,1]
        
        # Print intermediate values for debugging
        if not hasattr(self, '_debug_printed'):
            print("\n=== Forward Pass Debug ===")
            print(f"Final drift score: {drift_score.detach().cpu().numpy()}\n")
            self._debug_printed = True
        
        return drift_score

def train_model(
    data: DataSplit,
    batch_size: int = 32,
    epochs: int = 75,
    learning_rate: float = 0.0001,
    early_stopping_patience: int = 15,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
    distribution_weight: float = 0.1,  # Weight for distribution regularization
    target_std: float = 0.2,  # Target standard deviation for outputs
) -> Tuple[EnhancedTopicDriftDetector, Dict[str, list]]:
    """Train the model with advanced optimizations and augmentations."""
    # Set device and enable cuda optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print("\n=== Training Configuration ===")
    print(f"Device: {device}")
    print(f"Initial batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Max gradient norm: {max_grad_norm}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Distribution regularization weight: {distribution_weight}")
    print(f"Target standard deviation: {target_std}")

    # Create model save directory if it doesn't exist
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / "best_topic_drift_model.pt"

    # Create datasets with data on the appropriate device
    train_embeddings = data.train_embeddings.clone().to(device)
    train_labels = data.train_labels.clone().to(device)
    val_embeddings = data.val_embeddings.clone().to(device)
    val_labels = data.val_labels.clone().to(device)

    print("\n=== Dataset Information ===")
    print(f"Training set shape: {train_embeddings.shape}")
    print(f"Validation set shape: {val_embeddings.shape}")

    # Create data loaders
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
        persistent_workers=False
    )

    val_dataset = TensorDataset(val_embeddings, val_labels)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
        persistent_workers=False
    )

    # Initialize model and move to device
    embedding_dim = train_embeddings.shape[1] // 8
    model = EnhancedTopicDriftDetector(embedding_dim).to(device)
    
    criterion = nn.MSELoss()
    
    # Enhanced optimizer configuration
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Initialize metrics tracking
    metrics = {
        "train_losses": [],
        "train_rmse": [],
        "train_r2": [],
        "val_losses": [],
        "val_rmse": [],
        "val_r2": [],
        "learning_rates": [],
        "distribution_losses": [],
        "output_stds": []
    }

    # Initialize automatic mixed precision (AMP) scaler
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    # Learning rate scheduler with warmup
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # Initialize metrics
    rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    r2score = torchmetrics.R2Score().to(device)

    # Early stopping setup
    best_val_rmse = float("inf")
    patience_counter = 0
    best_model_state = None

    # Training loop with tqdm
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        dist_loss_sum = 0.0
        std_sum = 0.0
        rmse.reset()
        r2score.reset()

        batch_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")
        for batch_x, batch_y in batch_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(batch_x)
                base_loss = criterion(outputs, batch_y.unsqueeze(1))
                dist_loss = distribution_loss(outputs, target_std)
                loss = base_loss + distribution_weight * dist_loss

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()

            # Learning rate scheduling
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Update metrics
            train_loss += base_loss.item()
            dist_loss_sum += dist_loss.item()
            std_sum += outputs.std().item()
            rmse.update(outputs.view(-1), batch_y)
            r2score.update(outputs.view(-1), batch_y)
            
            # Display current batch metrics
            batch_pbar.set_postfix({
                "base_loss": f"{base_loss.item():.4f}",
                "dist_loss": f"{dist_loss.item():.4f}",
                "std": f"{outputs.std().item():.4f}",
                "rmse": f"{rmse.compute().item():.4f}",
                "lr": f"{current_lr:.2e}"
            })

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_dist_loss = dist_loss_sum / len(train_loader)
        avg_std = std_sum / len(train_loader)
        train_rmse_val = rmse.compute().item()
        train_r2_val = r2score.compute().item()
        
        train_results = {
            "loss": avg_train_loss,
            "rmse": train_rmse_val,
            "r2": train_r2_val,
        }
        
        # Update metrics
        metrics["train_losses"].append(train_results["loss"])
        metrics["train_rmse"].append(train_results["rmse"])
        metrics["train_r2"].append(train_results["r2"])
        metrics["distribution_losses"].append(avg_dist_loss)
        metrics["output_stds"].append(avg_std)
        metrics["learning_rates"].append(current_lr)

        print(f"\nEpoch {epoch+1} Training Metrics:")
        print(f"Loss: {avg_train_loss:.4f}")
        print(f"RMSE: {train_rmse_val:.4f}")
        print(f"R²: {train_r2_val:.4f}")
        print(f"Distribution Loss: {avg_dist_loss:.4f}")
        print(f"Output Std: {avg_std:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dist_loss = 0.0
        val_std = 0.0
        rmse.reset()
        r2score.reset()

        with torch.no_grad():
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    dist_loss = distribution_loss(outputs, target_std)
                    
                    val_loss += loss.item()
                    val_dist_loss += dist_loss.item()
                    val_std += outputs.std().item()
                    
                    rmse.update(outputs.view(-1), batch_y)
                    r2score.update(outputs.view(-1), batch_y)

        # Calculate validation metrics
        val_results = {
            "loss": val_loss / len(val_loader),
            "dist_loss": val_dist_loss / len(val_loader),
            "std": val_std / len(val_loader),
            "rmse": rmse.compute().item(),
            "r2": r2score.compute().item(),
        }

        # Update metrics
        metrics["val_losses"].append(val_results["loss"])
        metrics["val_rmse"].append(val_results["rmse"])
        metrics["val_r2"].append(val_results["r2"])

        # Early stopping check
        if val_results["rmse"] < best_val_rmse:
            best_val_rmse = val_results["rmse"]
            best_model_state = model.state_dict()
            
            # Save best model with additional training info
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_rmse': best_val_rmse,
                'train_rmse': train_results["rmse"],
                'hyperparameters': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_dim': 512,  # Fixed value from model init
                    'num_heads': model.num_heads,
                    'max_grad_norm': max_grad_norm,
                    'warmup_steps': warmup_steps,
                    'distribution_weight': distribution_weight,
                    'target_std': target_std
                }
            }, model_path)
            print(f"\nSaved best model with validation RMSE: {best_val_rmse:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Update progress bar
        epoch_pbar.set_postfix({
            "train_loss": f"{train_results['loss']:.4f}",
            "train_rmse": f"{train_results['rmse']:.4f}",
            "val_loss": f"{val_results['loss']:.4f}",
            "val_rmse": f"{val_results['rmse']:.4f}",
            "std": f"{val_results['std']:.4f}",
            "lr": f"{current_lr:.2e}"
        })

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model with validation RMSE: {best_val_rmse:.4f}")

    return model, metrics

def main():
    """Main training function."""
    print("\n=== Running Full Training ===")
    
    # Load and prepare data
    data = load_from_huggingface("leonvanbokhorst/topic-drift-v2")
    data = prepare_training_data(data, window_size=8, batch_size=32)
    
    # Train model
    model, metrics = train_model(data)
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(metrics["train_losses"], label="Train")
    plt.plot(metrics["val_losses"], label="Validation")
    plt.title("Loss")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics["train_rmse"], label="Train")
    plt.plot(metrics["val_rmse"], label="Validation")
    plt.title("RMSE")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics["train_r2"], label="Train")
    plt.plot(metrics["val_r2"], label="Validation")
    plt.title("R²")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(metrics["output_stds"], label="Output Std")
    plt.axhline(y=0.2, color='r', linestyle='--', label="Target Std")
    plt.title("Output Standard Deviation")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("models/training_curves.png")
    plt.close()

if __name__ == "__main__":
    main()