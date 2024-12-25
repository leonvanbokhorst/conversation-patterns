import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
from topic_drift.data_loader import load_from_huggingface
from topic_drift.data_prep import prepare_training_data, DataSplit
from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Linguistic markers for topic transitions
TRANSITION_MARKERS = [
    "so",           # Topic shift
    "anyway",       # Topic shiftA
    "by the way",   # Topic introduction
    "speaking of",  # Related topic
    "oh",          # Sudden realization/topic change
    "well",        # Hesitation/transition
    "actually",     # Contradiction/shift
    "but",         # Contrast/shift
    "however",      # Contrast/shift
    "meanwhile",    # Parallel topic
]

class EnhancedTopicDriftDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        """Initialize the topic drift detection model.

        Args:
            input_dim: Dimension of a single turn embedding
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.input_dim = input_dim
        self._has_printed_dims = False  # Add flag for dimension printing
        
        # Define attention parameters first
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        
        print("\n=== Model Architecture ===")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Number of attention heads: {self.num_heads}")
        print(f"Head dimension: {self.head_dim}")

        # Embedding processor with residual connection and increased dropout
        self.embedding_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.35),  # Increased dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.35)  # Increased dropout
        )

        # Dynamic multi-head attention with L2 regularization
        self.local_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, self.head_dim, bias=False),  # Removed bias for regularization
                nn.ReLU(),
                nn.Linear(self.head_dim, 1, bias=False),  # Removed bias for regularization
                nn.Softmax(dim=1)
            ) for _ in range(self.num_heads)
        ])
        
        # Global context attention with position encoding
        self.position_encoder = nn.Parameter(torch.randn(1, 8, hidden_dim))
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),  # Removed bias
            nn.Tanh(),
            nn.Dropout(0.35),  # Added dropout
            nn.Linear(hidden_dim, self.num_heads, bias=False),
            nn.Softmax(dim=1)
        )
        
        # Enhanced pattern detection with increased regularization
        self.pattern_detector = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            bidirectional=True,
            dropout=0.35,  # Increased dropout
            bias=False  # Removed bias for regularization
        )

        # Pattern classifier with regularization
        self.pattern_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, len(self.get_pattern_types()), bias=False),
            nn.Softmax(dim=-1)
        )

        # Dynamic weight generator with regularization
        self.weight_generator = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, 3, bias=False),
            nn.Softmax(dim=-1)
        )

        # Final regression with deeper network and increased regularization
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim // 2, 1, bias=False),
            nn.Sigmoid()
        )

    def get_pattern_types(self):
        return [
            "maintain",      # No drift
            "gentle_wave",   # Subtle topic evolution
            "single_peak",   # One clear transition
            "multi_peak",    # Multiple transitions
            "ascending",     # Gradually increasing drift
            "descending",    # Gradually decreasing drift
            "abrupt"        # Sudden topic change
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with multi-level attention."""
        batch_size = x.shape[0]
        window_size = 8
        
        # Only print dimensions on first forward pass
        if not self._has_printed_dims:
            print("\n=== Forward Pass Dimensions ===")
            print(f"Input shape: {x.shape}")
            
            # Reshape and process embeddings
            x = x.view(batch_size, window_size, self.input_dim)
            print(f"Reshaped input: {x.shape}")
            
            processed = self.embedding_processor(x)
            print(f"Processed embeddings: {processed.shape}")
            
            # Add positional encoding
            processed = processed + self.position_encoder
            print(f"After positional encoding: {processed.shape}")
            
            # Multi-head local attention
            local_attentions = []
            for head_idx, head in enumerate(self.local_attention):
                head_attn = head(processed)
                print(f"Head {head_idx} attention: {head_attn.shape}")
                local_attentions.append(head_attn)
            local_attention = torch.cat(local_attentions, dim=2)
            print(f"Combined local attention: {local_attention.shape}")
            
            # Global attention with position awareness
            global_attention = self.global_attention(processed)
            print(f"Global attention: {global_attention.shape}")
            
            # Enhanced pattern detection with proper batch handling
            lstm_input = processed.transpose(0, 1)
            pattern_output, (h_n, _) = self.pattern_detector(lstm_input)
            pattern_output = pattern_output.transpose(0, 1)
            print(f"Pattern detector output: {pattern_output.shape}")
            print(f"Pattern hidden state: {h_n.shape}")
            
            # Get last forward and backward states from last layer
            num_layers = 3
            num_directions = 2
            last_layer_forward = h_n[2 * num_layers - 2]
            last_layer_backward = h_n[2 * num_layers - 1]
            pattern_hidden = torch.cat([last_layer_forward, last_layer_backward], dim=1)
            print(f"Pattern hidden concatenated: {pattern_hidden.shape}")
            
            pattern_probs = self.pattern_classifier(pattern_hidden)
            print(f"Pattern probabilities: {pattern_probs.shape}")
            
            weights = self.weight_generator(pattern_hidden)
            print(f"Generated weights: {weights.shape}")
            
            local_context = local_attention.mean(dim=2)
            global_context = global_attention.mean(dim=2)
            pattern_context = pattern_output.mean(dim=2)
            print(f"Context shapes - Local: {local_context.shape}, Global: {global_context.shape}, Pattern: {pattern_context.shape}")
            
            contexts = torch.stack([local_context, global_context, pattern_context], dim=2)
            print(f"Stacked contexts: {contexts.shape}")
            
            weights = weights.unsqueeze(1)
            print(f"Expanded weights: {weights.shape}")
            
            attention = torch.bmm(contexts, weights.transpose(1, 2))
            attention = attention.softmax(dim=1)
            print(f"Final attention: {attention.shape}")
            
            transitions = self.semantic_bridge_detection(processed)
            print(f"Transitions: {transitions.shape}")
            
            context = torch.sum(attention * processed, dim=1)
            print(f"Final context: {context.shape}")
            
            base_score = self.regressor(context)
            print(f"Base score: {base_score.shape}")
            
            pattern_weights = torch.tensor(
                [0.8, 0.9, 1.0, 1.1, 1.2, 0.9, 1.3],
                device=pattern_probs.device
            ).unsqueeze(0)
            
            pattern_factor = torch.sum(pattern_probs * pattern_weights, dim=1, keepdim=True)
            transition_factor = 1 + transitions.mean(dim=1, keepdim=True)
            final_score = base_score * pattern_factor * transition_factor
            print(f"Final score: {final_score.shape}\n")
            
            self._has_printed_dims = True  # Set flag to True after first print
        else:
            # Regular forward pass without printing
            x = x.view(batch_size, window_size, self.input_dim)
            processed = self.embedding_processor(x)
            processed = processed + self.position_encoder
            
            local_attentions = []
            for head in self.local_attention:
                head_attn = head(processed)
                local_attentions.append(head_attn)
            local_attention = torch.cat(local_attentions, dim=2)
            
            global_attention = self.global_attention(processed)
            
            lstm_input = processed.transpose(0, 1)
            pattern_output, (h_n, _) = self.pattern_detector(lstm_input)
            pattern_output = pattern_output.transpose(0, 1)
            
            last_layer_forward = h_n[2 * 3 - 2]
            last_layer_backward = h_n[2 * 3 - 1]
            pattern_hidden = torch.cat([last_layer_forward, last_layer_backward], dim=1)
            
            pattern_probs = self.pattern_classifier(pattern_hidden)
            weights = self.weight_generator(pattern_hidden)
            
            local_context = local_attention.mean(dim=2)
            global_context = global_attention.mean(dim=2)
            pattern_context = pattern_output.mean(dim=2)
            
            contexts = torch.stack([local_context, global_context, pattern_context], dim=2)
            weights = weights.unsqueeze(1)
            
            attention = torch.bmm(contexts, weights.transpose(1, 2))
            attention = attention.softmax(dim=1)
            
            transitions = self.semantic_bridge_detection(processed)
            context = torch.sum(attention * processed, dim=1)
            base_score = self.regressor(context)
            
            pattern_weights = torch.tensor(
                [0.8, 0.9, 1.0, 1.1, 1.2, 0.9, 1.3],
                device=pattern_probs.device
            ).unsqueeze(0)
            
            pattern_factor = torch.sum(pattern_probs * pattern_weights, dim=1, keepdim=True)
            transition_factor = 1 + transitions.mean(dim=1, keepdim=True)
            final_score = base_score * pattern_factor * transition_factor
            
        return torch.clamp(final_score, 0, 1)

    def semantic_bridge_detection(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Detect semantic bridges between turns."""
        batch_size, seq_len, _ = embeddings.shape
        
        # Pairwise semantic similarity
        similarities = torch.bmm(
            embeddings, embeddings.transpose(1, 2)
        )
        
        # Detect transition patterns
        transitions = torch.zeros(batch_size, seq_len - 1, device=embeddings.device)
        
        # Create transition scores based on similarity thresholds
        high_sim = (similarities[:, range(seq_len-1), range(1, seq_len)] > 0.8).float()
        med_sim = (similarities[:, range(seq_len-1), range(1, seq_len)] > 0.6).float()
        low_sim = (similarities[:, range(seq_len-1), range(1, seq_len)] > 0.4).float()
        
        # Apply transition scores using logical operations on tensors
        transitions = (
            0.11 * high_sim +  # topic_maintenance
            0.15 * (med_sim * (1 - high_sim)) +  # smooth_transition
            0.18 * (low_sim * (1 - med_sim)) +  # topic_shift
            0.21 * (1 - low_sim)  # abrupt_change
        )
            
        return transitions



def train_model(
    data: DataSplit,
    batch_size: int = 32,
    epochs: int = 75,  # Increased epochs
    learning_rate: float = 0.0001,
    early_stopping_patience: int = 15,  # Increased patience
) -> Tuple[EnhancedTopicDriftDetector, Dict[str, list]]:
    """Train the topic drift detection model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n=== Training Configuration ===")
    print(f"Device: {device}")
    print(f"Initial batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")

    # Create model save directory if it doesn't exist
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / "best_topic_drift_model.pt"

    # Move data to device and ensure batch size compatibility
    train_embeddings = data.train_embeddings.to(device)
    train_labels = data.train_labels.to(device)
    val_embeddings = data.val_embeddings.to(device)
    val_labels = data.val_labels.to(device)

    print("\n=== Dataset Information ===")
    print(f"Training set shape: {train_embeddings.shape}")
    print(f"Validation set shape: {val_embeddings.shape}")

    # Adjust batch size to ensure it divides dataset size
    train_size = len(train_embeddings)
    val_size = len(val_embeddings)
    
    # Calculate batch size that divides both train and val sizes
    max_batch = min(batch_size, train_size // 2, val_size // 2)
    adjusted_batch = max_batch
    while train_size % adjusted_batch != 0 or val_size % adjusted_batch != 0:
        adjusted_batch -= 1
    batch_size = adjusted_batch
    print(f"Adjusted batch size to: {batch_size}")

    # Create data loaders with fixed batch size
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )

    val_dataset = TensorDataset(val_embeddings, val_labels)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        drop_last=True
    )

    # Initialize model and training components
    embedding_dim = data.train_embeddings.shape[1] // 8
    model = EnhancedTopicDriftDetector(embedding_dim).to(device)
    criterion = nn.MSELoss()
    
    # Add L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # Initialize metrics
    rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    r2score = torchmetrics.R2Score().to(device)

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
        train_loss = 0.0
        rmse.reset()
        r2score.reset()

        batch_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")
        for batch_x, batch_y in batch_pbar:
            # Forward pass
            outputs = model(batch_x)  # Shape: [batch_size, 1]
            loss = criterion(outputs, batch_y.unsqueeze(1))  # Ensure target has shape [batch_size, 1]

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics - keep dimensions consistent
            train_loss += loss.item()
            rmse.update(outputs.view(-1), batch_y)  # Use view instead of squeeze to maintain dimensions
            r2score.update(outputs.view(-1), batch_y)
            
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # Calculate training metrics
        train_results = {
            "loss": train_loss / len(train_loader),
            "rmse": rmse.compute().item(),
            "r2": r2score.compute().item(),
        }

        # Validation phase
        model.eval()
        val_loss = 0.0
        rmse.reset()
        r2score.reset()

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                val_loss += loss.item()
                
                rmse.update(outputs.view(-1), batch_y)
                r2score.update(outputs.view(-1), batch_y)

        # Calculate validation metrics
        val_results = {
            "loss": val_loss / len(val_loader),
            "rmse": rmse.compute().item(),
            "r2": r2score.compute().item(),
        }

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
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': best_val_rmse,
                'train_rmse': train_results["rmse"],
                'hyperparameters': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_dim': model.embedding_processor[0].out_features,
                    'num_heads': model.num_heads
                }
            }, model_path)
            print(f"\nSaved best model with validation RMSE: {best_val_rmse:.4f}")
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
        print(f"\nRestored best model with validation RMSE: {best_val_rmse:.4f}")

    return model, metrics


def evaluate_model(
    model: EnhancedTopicDriftDetector,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
    r2score = torchmetrics.R2Score().to(device)
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            total_loss += loss.item()

            rmse.update(outputs.view(-1), batch_y)
            r2score.update(outputs.view(-1), batch_y)

    return {
        "loss": total_loss / len(dataloader),
        "rmse": rmse.compute().item(),
        "r2": r2score.compute().item(),
    }


def plot_training_curves(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot training and validation metrics over epochs.
    
    Args:
        metrics: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(131)
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot RMSE
    plt.subplot(132)
    plt.plot(metrics['train_rmse'], label='Train RMSE')
    plt.plot(metrics['val_rmse'], label='Val RMSE')
    plt.title('RMSE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    # Plot R²
    plt.subplot(133)
    plt.plot(metrics['train_r2'], label='Train R²')
    plt.plot(metrics['val_r2'], label='Val R²')
    plt.title('R² Score Curves')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_attention(
    model: EnhancedTopicDriftDetector,
    sample_windows: torch.Tensor,
    sample_texts: List[List[str]],
    sample_scores: torch.Tensor,
    device: torch.device,
    save_path: str = None
):
    """Visualize attention weights for sample windows."""
    model.eval()
    with torch.no_grad():
        batch_size = sample_windows.shape[0]
        window_size = 8
        
        # Get model attention weights and predictions
        x = sample_windows.to(device)
        x = x.view(batch_size, window_size, model.input_dim)
        processed = model.embedding_processor(x)
        processed = processed + model.position_encoder
        
        # Get multi-head local attention
        local_attentions = []
        for head in model.local_attention:
            head_attn = head(processed)
            local_attentions.append(head_attn)
        local_attention = torch.cat(local_attentions, dim=2)
        
        # Get global attention
        global_attention = model.global_attention(processed)
        
        # Get pattern detection
        pattern_output, (h_n, _) = model.pattern_detector(processed)
        
        # Get pattern weights
        pattern_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        weights = model.weight_generator(pattern_hidden)
        weights = weights.unsqueeze(1).expand(-1, window_size, -1)
        
        # Combine attention mechanisms
        attention_weights = (
            weights[:, :, 0:1] * local_attention.mean(dim=2, keepdim=True) +
            weights[:, :, 1:2] * global_attention.mean(dim=2, keepdim=True) +
            weights[:, :, 2:3] * pattern_output.mean(dim=2, keepdim=True)
        )
        
        attention_weights = attention_weights.cpu().numpy()
        predictions = model(sample_windows.to(device)).squeeze().cpu().numpy()
        
        # Plot attention heatmaps
        plt.figure(figsize=(15, 4 * batch_size))
        for i in range(batch_size):
            plt.subplot(batch_size, 1, i + 1)
            
            # Create heatmap
            sns.heatmap(
                attention_weights[i].T,
                cmap='YlOrRd',
                xticklabels=[f"Turn {j+1}\n{sample_texts[i][j][:50]}..." for j in range(window_size)],
                yticklabels=False,
                cbar_kws={'label': 'Attention Weight'}
            )
            
            # Rotate x-labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add title with scores
            plt.title(
                f'Window {i+1} - Predicted Drift: {predictions[i]:.4f}, '
                f'Actual Drift: {sample_scores[i].item():.4f}\n'
                f'Attention Distribution Across Conversation Turns',
                pad=20
            )
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def plot_prediction_distribution(
    model: EnhancedTopicDriftDetector,
    test_loader: DataLoader,
    device: torch.device,
    save_path: str = None
):
    """Plot distribution of predictions vs actual values.
    
    Args:
        model: Trained EnhancedTopicDriftDetector model
        test_loader: DataLoader for test data
        device: Device to run model on
        save_path: Optional path to save the plot
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(121)
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Drift Score')
    plt.ylabel('Predicted Drift Score')
    plt.title('Predictions vs Actual Values')
    plt.grid(True)
    
    # Error distribution
    plt.subplot(122)
    errors = np.array(all_preds) - np.array(all_targets)
    plt.hist(errors, bins=50, density=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


class TransitionPatternModule(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Pattern weights based on our analysis
        self.pattern_weights = {
            "gentle_wave": (0.10, 0.16),    # Low drift range
            "single_peak": (0.13, 0.19),    # Medium drift range
            "ascending_stairs": (0.16, 0.22) # High drift range
        }
        
        # Pattern detection layers
        self.pattern_detector = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True
        )


def train_with_patterns(
    model: EnhancedTopicDriftDetector,
    data: DataSplit,
    batch_size: int = 32,
    epochs: int = 20,
) -> None:
    """Enhanced training with pattern recognition."""
    # Additional loss components
    pattern_loss = nn.MSELoss()  # For pattern detection
    transition_loss = nn.L1Loss() # For transition smoothness
    
    # Combined loss function
    def combined_loss(pred, target, patterns, transitions):
        base_loss = nn.MSELoss()(pred, target)
        pattern_penalty = pattern_loss(patterns, expected_patterns)
        transition_smoothness = transition_loss(transitions, expected_transitions)
        
        return (
            0.6 * base_loss +          # Base drift prediction
            0.2 * pattern_penalty +     # Pattern adherence
            0.2 * transition_smoothness # Transition smoothness
        )


def main():
    """Load data, prepare it, and train the model."""
    print("\n=== Running Full Training ===")

    # Load conversation data from Hugging Face
    conversation_data = load_from_huggingface()

    # Prepare training data with splits
    data = prepare_training_data(
        conversation_data,
        window_size=8,
        batch_size=32,
        max_workers=16,
        force_recompute=True,  # Use cached data if available
    )

    # Train model with updated parameters
    model, metrics = train_model(
        data,
        batch_size=32,  # Reduced batch size for better generalization
        epochs=75,      # Increased epochs
        learning_rate=0.0001,
        early_stopping_patience=15,  # Increased patience
    )

    # Get device
    device = next(model.parameters()).device

    # Create test dataset with same batch size as training
    test_dataset = TensorDataset(data.test_embeddings.to(device), data.test_labels.to(device))
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        drop_last=True
    )

    # Plot training curves
    plot_training_curves(metrics, save_path="models/full_training_curves.png")

    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, device)

    # Print final metrics
    print("\nFull Training Results:")
    print(f"Best Validation RMSE: {min(metrics['val_rmse']):.4f}")
    print(f"Best Validation R²: {max(metrics['val_r2']):.4f}")
    print("\nTest Set Results:")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"RMSE: {test_results['rmse']:.4f}")
    print(f"R²: {test_results['r2']:.4f}")

    # Save final metrics
    metrics_path = Path("models") / "full_training_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("=== Full Training Results ===\n")
        f.write(f"Best Validation RMSE: {min(metrics['val_rmse']):.4f}\n")
        f.write(f"Best Validation R²: {max(metrics['val_r2']):.4f}\n")
        f.write("\n=== Test Set Results ===\n")
        f.write(f"Loss: {test_results['loss']:.4f}\n")
        f.write(f"RMSE: {test_results['rmse']:.4f}\n")
        f.write(f"R²: {test_results['r2']:.4f}\n")


if __name__ == "__main__":
    main()