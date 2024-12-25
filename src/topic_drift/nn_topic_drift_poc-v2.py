import torch
import torch.nn as nn
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
import torch.nn.functional as F
import math
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer warnings
torch.multiprocessing.set_sharing_strategy('file_system')  # Fix multiprocessing issues

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

class PreNorm(nn.Module):
    """Pre-normalization module for transformer-style attention."""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with key/query/value projections."""
    def __init__(self, dim: int, num_heads: int = 4, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        inner_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        # Separate projections for key, query, value
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape  # batch, sequence length, dimension
        h = self.num_heads

        # Project and reshape for multi-head attention
        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(x).view(b, n, h, -1).transpose(1, 2)
        v = self.to_v(x).view(b, n, h, -1).transpose(1, 2)

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        # Combine heads
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Using GELU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalPatternDetector(nn.Module):
    """Hierarchical pattern detection with multi-scale analysis."""
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_dim,  # Keep input size constant
                hidden_size=hidden_dim,  # Keep hidden size constant
                bidirectional=True,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.downsample = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim * 2,  # Bidirectional output
                out_channels=hidden_dim,     # Project back to hidden_dim
                kernel_size=2,
                stride=2
            ) for _ in range(num_layers - 1)
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch_size = x.size(0)
        outputs = []
        current = x
        
        for i, (lstm, down) in enumerate(zip(self.layers[:-1], self.downsample)):
            # LSTM processing
            lstm_out, _ = lstm(current)  # [batch, seq_len, hidden_dim * 2]
            outputs.append(lstm_out)
            
            # Downsample and project back to hidden_dim
            current = down(lstm_out.transpose(1, 2)).transpose(1, 2)
        
        # Final LSTM layer without downsampling
        final_out, _ = self.layers[-1](current)
        outputs.append(final_out)
        
        return outputs


class TransitionDetector(nn.Module):
    """Explicit transition point detection with linguistic marker attention."""
    def __init__(self, hidden_dim: int, num_markers: int = len(TRANSITION_MARKERS)):
        super().__init__()
        self.marker_embeddings = nn.Parameter(torch.randn(num_markers, hidden_dim))
        self.marker_attention = MultiHeadAttention(hidden_dim, num_heads=4)
        self.transition_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expand marker embeddings for batch processing
        batch_size = x.size(0)
        markers = self.marker_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute attention between input and markers
        marker_context = self.marker_attention(markers)  # [batch, num_markers, hidden_dim]
        
        # Compute transition scores
        transitions = []
        for i in range(x.size(1) - 1):
            current = x[:, i:i+2].mean(dim=1)  # Average consecutive turns
            # Concatenate with marker context
            combined = torch.cat([
                current,
                marker_context.mean(dim=1)  # Average marker context
            ], dim=-1)
            score = self.transition_scorer(combined)
            transitions.append(score)
            
        transition_scores = torch.stack(transitions, dim=1)  # [batch, seq_len-1, 1]
        return transition_scores, marker_context


class PatternSelfAttention(nn.Module):
    """Self-attention mechanism for pattern interaction."""
    def __init__(self, hidden_dim: int, num_patterns: int):
        super().__init__()
        self.pattern_embeddings = nn.Parameter(torch.randn(num_patterns, hidden_dim))
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads=4)
        self.pattern_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pattern_logits: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = pattern_logits.size(0)
        
        # Get pattern-weighted embeddings
        pattern_weights = F.softmax(pattern_logits, dim=-1)
        weighted_patterns = torch.matmul(
            pattern_weights,
            self.pattern_embeddings
        )
        
        # Self-attention between patterns
        attended_patterns = self.self_attention(weighted_patterns.unsqueeze(1))
        
        # Compute gating mechanism
        gate_input = torch.cat([
            attended_patterns.squeeze(1),
            hidden_states.mean(dim=1)
        ], dim=-1)
        gate = self.pattern_gate(gate_input)
        
        # Apply gating
        gated_patterns = gate * pattern_weights
        return gated_patterns


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

        # Enhanced embedding processor with residual connection
        self.embedding_processor = nn.Sequential(
            PreNorm(input_dim, 
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.35),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ),
            nn.Dropout(0.35)
        )

        # Enhanced attention blocks with residual connections
        self.attention_blocks = nn.ModuleList([
            nn.ModuleList([
                PreNorm(hidden_dim, 
                    MultiHeadAttention(hidden_dim, self.num_heads, self.head_dim, dropout=0.35)),
                PreNorm(hidden_dim, 
                    FeedForward(hidden_dim, hidden_dim * 4, dropout=0.35))
            ]) for _ in range(3)  # 3 layers of attention
        ])
        
        # Position encoding with learned parameters
        self.position_encoder = nn.Parameter(torch.randn(1, 8, hidden_dim))
        
        # Hierarchical pattern detection
        self.hierarchical_detector = HierarchicalPatternDetector(hidden_dim)
        
        # Transition detection with linguistic markers
        self.transition_detector = TransitionDetector(hidden_dim)
        
        # Dimension adapter for pattern features
        self.pattern_dim_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),  # 1536 -> 1536 (keep dims consistent)
            nn.GELU(),
            nn.Dropout(0.35)
        )
        
        # Pattern classifier with proper input dimension
        self.pattern_classifier = nn.Sequential(
            PreNorm(hidden_dim * 3,  # Now matches the adapted dimensions
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Dropout(0.35),
                    nn.Linear(hidden_dim, len(self.get_pattern_types()), bias=False)
                )
            )
        )
        
        # Pattern interaction with self-attention
        self.pattern_interaction = PatternSelfAttention(
            hidden_dim,
            num_patterns=len(self.get_pattern_types())
        )

        # Final regression with residual connections
        self.regressor = nn.Sequential(
            PreNorm(hidden_dim * 3 + 1,  # Changed to match combined features dimensions
                nn.Sequential(
                    nn.Linear(hidden_dim * 3 + 1, hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Dropout(0.35),
                    nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
                )
            ),
            PreNorm(hidden_dim // 2,
                nn.Sequential(
                    nn.GELU(),
                    nn.Dropout(0.35),
                    nn.Linear(hidden_dim // 2, 1, bias=False),
                    nn.Sigmoid()
                )
            )
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
        b = x.shape[0]  # batch size
        
        # Process embeddings first to match hidden dimension
        x = x.view(b, 8, -1)  # [batch, seq_len, input_dim]
        x = self.embedding_processor(x)  # Now x is [batch, seq_len, hidden_dim]
        
        # Add position encoding after embedding processing
        x = x + self.position_encoder
        
        # Apply attention blocks with residual connections
        for attn, ff in self.attention_blocks:
            x = attn(x) + x
            x = ff(x) + x
        
        # Hierarchical pattern detection
        pattern_scales = self.hierarchical_detector(x)  # List of tensors [batch, seq_len, hidden_dim * 2]
        
        # Transition detection
        transition_scores, marker_context = self.transition_detector(x)  # [batch, seq_len-1, 1], [batch, num_markers, hidden_dim]
        
        # Pattern classification - ensure proper dimensions
        pattern_features = torch.cat([
            pattern_scales[-1].mean(dim=1),  # [batch, hidden_dim * 2]
            marker_context.mean(dim=1)       # [batch, hidden_dim]
        ], dim=-1)  # Result: [batch, hidden_dim * 3]
        
        # Adapt pattern features dimensions
        pattern_features = self.pattern_dim_adapter(pattern_features)  # [batch, hidden_dim * 3]
        
        # Pattern classification
        pattern_logits = self.pattern_classifier(pattern_features)  # [batch, num_patterns]
        
        # Pattern interaction
        pattern_weights = self.pattern_interaction(pattern_logits, x)  # [batch, num_patterns]
        
        # Combine all features for final prediction
        combined = torch.cat([
            x.mean(dim=1),                      # [batch, hidden_dim]
            pattern_scales[-1].mean(dim=1),     # [batch, hidden_dim * 2]
            transition_scores.mean(dim=1)        # [batch, 1]
        ], dim=-1)  # Result: [batch, hidden_dim * 3 + 1]
        
        # Final prediction
        out = self.regressor(combined)  # [batch, 1]
        
        return out

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


class DynamicWindowAugmentation:
    """Dynamic window size augmentation with adaptive padding/truncation."""
    def __init__(self, base_window_size: int = 8, size_range: int = 2):
        self.base_size = base_window_size
        self.size_range = size_range
    
    def __call__(self, conversation: torch.Tensor) -> torch.Tensor:
        # Keep original shape for later
        batch_size = conversation.shape[0]
        total_dim = conversation.shape[1]
        dim = total_dim // self.base_size
        
        # Reshape input from [batch_size, base_size * dim] to [batch_size, base_size, dim]
        x = conversation.view(batch_size, self.base_size, dim)
        
        # Randomly choose window size
        new_size = self.base_size + torch.randint(-self.size_range, self.size_range + 1, (1,)).item()
        
        if new_size < self.base_size:
            # Random truncation
            start_idx = torch.randint(0, self.base_size - new_size + 1, (1,)).item()
            x = x[:, start_idx:start_idx + new_size, :]
            # Pad back to original size
            padding = self.base_size - new_size
            x = F.pad(x, (0, 0, 0, padding), mode='replicate')
        elif new_size > self.base_size:
            # Adaptive padding using interpolation
            x = F.interpolate(
                x.transpose(1, 2),  # [B, D, T]
                size=new_size,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [B, T, D]
            # Truncate back to original size
            x = x[:, :self.base_size, :]
        
        # Return to original shape [batch_size, base_size * dim]
        return x.reshape(batch_size, total_dim)


class ContrastiveLearningLoss(nn.Module):
    """Contrastive learning with pattern-aware positive/negative sampling."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8  # For numerical stability
        
    def forward(
        self,
        embeddings: torch.Tensor,
        pattern_weights: torch.Tensor,
        transition_scores: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarity matrix with numerical stability
        sim_matrix = torch.clamp(
            torch.matmul(embeddings, embeddings.transpose(0, 1)) / self.temperature,
            min=-10.0,  # Prevent extreme negative values
            max=10.0    # Prevent extreme positive values
        )
        
        # Create pattern-based positive pairs with normalization
        pattern_weights = F.normalize(pattern_weights, p=2, dim=-1)
        pattern_sim = torch.clamp(
            torch.matmul(pattern_weights, pattern_weights.transpose(0, 1)),
            min=-10.0,
            max=10.0
        )
        
        # Compute transition similarity matrix
        # First, get mean transition score per sequence
        mean_transitions = transition_scores.mean(dim=1).squeeze(-1)  # [batch_size]
        # Add small epsilon to prevent division by zero
        mean_transitions = mean_transitions + self.eps
        # Compute pairwise differences and normalize
        transition_diffs = torch.abs(mean_transitions.unsqueeze(0) - mean_transitions.unsqueeze(1))
        max_diff = torch.max(transition_diffs) + self.eps
        transition_sim = torch.clamp(
            1 - (transition_diffs / max_diff),
            min=0.0,
            max=1.0
        )
        
        # Combine similarities with weighted average
        combined_sim = (
            0.4 * sim_matrix +
            0.4 * pattern_sim +
            0.2 * transition_sim
        )
        
        # Mask out self-contrast cases
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        combined_sim.masked_fill_(mask, -10.0)  # Use finite value instead of -inf
        
        # Compute contrastive loss with numerical stability
        log_softmax = F.log_softmax(combined_sim, dim=-1)
        loss = -torch.mean(
            torch.clamp(
                torch.diagonal(log_softmax),
                min=-10.0,
                max=0.0
            )
        )
        
        return torch.clamp(loss, min=0.0, max=10.0)  # Ensure loss is finite and positive


class AdversarialTraining:
    """Adversarial training with pattern-aware perturbations."""
    def __init__(self, epsilon: float = 0.01, alpha: float = 0.005, steps: int = 3):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def perturb(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        pattern_weights: torch.Tensor,
        criterion: nn.Module
    ) -> torch.Tensor:
        # Initialize perturbation
        delta = torch.zeros_like(embeddings, requires_grad=True)
        
        # Pattern-aware perturbation
        pattern_importance = pattern_weights.mean(dim=-1, keepdim=True)
        
        for _ in range(self.steps):
            # Forward pass
            perturbed = embeddings + delta
            outputs = model(perturbed)
            # Use outputs directly for loss calculation
            loss = criterion(outputs, outputs.detach())  # Self-distillation loss
            
            # Compute gradients
            grad = torch.autograd.grad(loss, delta)[0]
            
            # Update perturbation
            delta = delta + self.alpha * grad.sign() * pattern_importance
            
            # Project to epsilon ball
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = torch.clamp(embeddings + delta, 0, 1) - embeddings
        
        return delta.detach()


def train_model(
    data: DataSplit,
    batch_size: int = 32,
    epochs: int = 75,
    learning_rate: float = 0.0001,
    early_stopping_patience: int = 15,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
    contrastive_weight: float = 0.1,
    adversarial_weight: float = 0.1
) -> Tuple[EnhancedTopicDriftDetector, Dict[str, list]]:
    """Train the model with advanced optimizations and augmentations."""
    # Set device and enable cuda optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better performance
        torch.backends.cudnn.allow_tf32 = True
    
    print("\n=== Training Configuration ===")
    print(f"Device: {device}")
    print(f"Initial batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Max gradient norm: {max_grad_norm}")
    print(f"Warmup steps: {warmup_steps}")

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

    # Create data loaders with fixed batch size and pin memory for faster GPU transfer
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        pin_memory=False,  # Data is already on GPU
        num_workers=0,  # Disable multi-processing for now
        persistent_workers=False
    )

    val_dataset = TensorDataset(val_embeddings, val_labels)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        drop_last=True,
        pin_memory=False,  # Data is already on GPU
        num_workers=0,  # Disable multi-processing for now
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
    
    # Initialize automatic mixed precision (AMP) scaler
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')  # Only enable for CUDA

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

    # Initialize metrics tracking with learning rates
    metrics = {
        "train_losses": [],
        "train_rmse": [],
        "train_r2": [],
        "val_losses": [],
        "val_rmse": [],
        "val_r2": [],
        "learning_rates": []
    }

    # Early stopping setup
    best_val_rmse = float("inf")
    patience_counter = 0
    best_model_state = None

    # Initialize augmentation and training components
    window_aug = DynamicWindowAugmentation()
    contrastive_loss = ContrastiveLearningLoss()
    adversarial = AdversarialTraining()

    # Training loop with tqdm
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        rmse.reset()
        r2score.reset()

        batch_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")
        for batch_x, batch_y in batch_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Apply dynamic window augmentation
            batch_x_aug = window_aug(batch_x)

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                # Original forward pass
                outputs = model(batch_x)
                base_loss = criterion(outputs, batch_y.unsqueeze(1))

                # Process augmented input
                batch_x_aug = batch_x_aug.view(batch_x_aug.size(0), 8, -1)  # [batch, seq_len, input_dim]
                batch_x_aug = model.embedding_processor(batch_x_aug)  # [batch, seq_len, hidden_dim]
                batch_x_aug = batch_x_aug + model.position_encoder  # Add position encoding

                # Apply attention blocks
                for attn, ff in model.attention_blocks:
                    batch_x_aug = attn(batch_x_aug) + batch_x_aug
                    batch_x_aug = ff(batch_x_aug) + batch_x_aug

                # Get pattern information
                pattern_scales = model.hierarchical_detector(batch_x_aug)  # Now input is properly processed
                transition_scores, marker_context = model.transition_detector(batch_x_aug)
                pattern_features = torch.cat([
                    pattern_scales[-1].mean(dim=1),  # [batch, hidden_dim * 2]
                    marker_context.mean(dim=1)       # [batch, hidden_dim]
                ], dim=-1)  # Result: [batch, hidden_dim * 3]

                # Adapt pattern features dimensions
                pattern_features = model.pattern_dim_adapter(pattern_features)  # [batch, hidden_dim * 3]

                # Pattern classification
                pattern_logits = model.pattern_classifier(pattern_features)

                # Pattern interaction
                pattern_weights = model.pattern_interaction(pattern_logits, batch_x_aug)

                # Contrastive learning (scaled down)
                contrast_loss = 0.01 * contrastive_loss(
                    batch_x_aug.mean(dim=1),
                    pattern_weights,
                    transition_scores
                )

                # Adversarial training (scaled down)
                delta = adversarial.perturb(model, batch_x, pattern_weights, criterion)
                adv_outputs = model(batch_x + delta)
                adv_loss = 0.01 * criterion(adv_outputs, batch_y.unsqueeze(1))

                # Combined loss
                loss = base_loss + contrast_loss + adv_loss

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
            rmse.update(outputs.view(-1), batch_y)
            r2score.update(outputs.view(-1), batch_y)
            
            # Display current batch metrics
            batch_pbar.set_postfix({
                "base_loss": f"{base_loss.item():.4f}",
                "total_loss": f"{loss.item():.4f}",
                "rmse": f"{rmse.compute().item():.4f}",
                "r2": f"{r2score.compute().item():.4f}",
                "lr": f"{current_lr:.2e}"
            })

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_rmse_val = rmse.compute().item()
        train_r2_val = r2score.compute().item()
        
        train_results = {
            "loss": avg_train_loss,
            "rmse": train_rmse_val,
            "r2": train_r2_val,
        }
        
        print(f"\nEpoch {epoch+1} Training Metrics:")
        print(f"Loss: {avg_train_loss:.4f}")
        print(f"RMSE: {train_rmse_val:.4f}")
        print(f"R²: {train_r2_val:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        rmse.reset()
        r2score.reset()

        with torch.no_grad():
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
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
        metrics["learning_rates"].append(current_lr)

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
                    'warmup_steps': warmup_steps
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


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


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

    # Prepare training data with splits, using cached embeddings
    data = prepare_training_data(
        conversation_data,
        window_size=8,
        batch_size=32,
        max_workers=16,
        force_recompute=False,  # Use cached data if available
    )

    # Train model with updated parameters
    model, metrics = train_model(
        data,
        batch_size=32,  # Reduced batch size for better generalization
        epochs=75,      # Increased epochs
        learning_rate=0.0001,
        early_stopping_patience=15,  # Increased patience
        max_grad_norm=1.0,  # For gradient clipping
        warmup_steps=100,     # Warmup steps for learning rate
        contrastive_weight=0.1,
        adversarial_weight=0.1
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