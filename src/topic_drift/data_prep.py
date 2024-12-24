"""Module for preparing conversation data for training."""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, NamedTuple
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
import json

from topic_drift.data_types import ConversationData
from topic_drift.llm_wrapper import OllamaWrapper


class DataSplit(NamedTuple):
    """Container for train/val/test split tensors."""

    train_embeddings: torch.Tensor
    train_labels: torch.Tensor
    val_embeddings: torch.Tensor
    val_labels: torch.Tensor
    test_embeddings: torch.Tensor
    test_labels: torch.Tensor


def split_data(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    """Split data into train, validation, and test sets.

    Args:
        embeddings: Full embeddings tensor
        labels: Full labels tensor
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        DataSplit object containing train/val/test tensors
    """
    # First split off test set
    train_val_emb, test_emb, train_val_labels, test_labels = train_test_split(
        embeddings.numpy(),
        labels.numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=labels.numpy(),
    )

    # Then split remaining data into train and validation
    train_emb, val_emb, train_labels, val_labels = train_test_split(
        train_val_emb,
        train_val_labels,
        test_size=val_size / (1 - test_size),  # Adjust for remaining data
        random_state=random_state,
        stratify=train_val_labels,
    )

    return DataSplit(
        train_embeddings=torch.from_numpy(train_emb).float(),
        train_labels=torch.from_numpy(train_labels).float(),
        val_embeddings=torch.from_numpy(val_emb).float(),
        val_labels=torch.from_numpy(val_labels).float(),
        test_embeddings=torch.from_numpy(test_emb).float(),
        test_labels=torch.from_numpy(test_labels).float(),
    )


def save_to_cache(
    data_split: DataSplit,
    cache_key: str,
) -> None:
    """Save prepared data splits to cache.

    Args:
        data_split: DataSplit object containing all splits
        cache_key: Cache key for the data
    """
    cache_dir = get_cache_path()
    np.savez(
        cache_dir / f"{cache_key}.npz",
        train_embeddings=data_split.train_embeddings.numpy(),
        train_labels=data_split.train_labels.numpy(),
        val_embeddings=data_split.val_embeddings.numpy(),
        val_labels=data_split.val_labels.numpy(),
        test_embeddings=data_split.test_embeddings.numpy(),
        test_labels=data_split.test_labels.numpy(),
    )


def load_from_cache(
    cache_key: str,
) -> Optional[DataSplit]:
    """Load prepared data splits from cache.

    Args:
        cache_key: Cache key for the data

    Returns:
        DataSplit object if cache exists, None otherwise
    """
    cache_path = get_cache_path() / f"{cache_key}.npz"
    if not cache_path.exists():
        return None

    print(f"Loading prepared data from cache: {cache_path}")
    data = np.load(cache_path)
    return DataSplit(
        train_embeddings=torch.from_numpy(data["train_embeddings"]).float(),
        train_labels=torch.from_numpy(data["train_labels"]).float(),
        val_embeddings=torch.from_numpy(data["val_embeddings"]).float(),
        val_labels=torch.from_numpy(data["val_labels"]).float(),
        test_embeddings=torch.from_numpy(data["test_embeddings"]).float(),
        test_labels=torch.from_numpy(data["test_labels"]).float(),
    )


@dataclass
class TurnWindow:
    """Container for a window of turns and their embeddings."""

    turns: List[str]  # List of consecutive turns in the window
    embeddings: List[np.ndarray] = None  # Embeddings for each turn
    drift_score: float = None  # Continuous drift score between 0 and 1
    window_similarity: float = None  # Average similarity within window


def get_cache_path() -> Path:
    """Get the path to the cache directory."""
    cache_dir = Path.home() / ".cache" / "topic_drift" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(
    conversation_data: ConversationData,
    window_size: int,
) -> str:
    """Generate a cache key for the prepared data.

    Args:
        conversation_data: The conversation data
        window_size: Size of the sliding window

    Returns:
        A unique cache key based on the data content and parameters
    """
    data_str = json.dumps(
        [conv["turns"] for conv in conversation_data.conversations], sort_keys=True
    )
    param_str = f"window_{window_size}"
    return hashlib.md5(f"{data_str}{param_str}".encode()).hexdigest()


async def process_window(
    window: TurnWindow,
    ollama: OllamaWrapper,
    executor: ThreadPoolExecutor,
) -> TurnWindow:
    """Process a window of turns asynchronously.

    Args:
        window: TurnWindow object containing turns
        ollama: OllamaWrapper instance
        executor: ThreadPoolExecutor for running embeddings in parallel

    Returns:
        Processed TurnWindow with embeddings and drift score
    """
    loop = asyncio.get_event_loop()

    # Get embeddings for all turns in parallel
    async def get_embedding(text: str) -> np.ndarray:
        return await loop.run_in_executor(executor, ollama.get_embeddings, text)

    # Process all embeddings in parallel
    embedding_tasks = [get_embedding(turn) for turn in window.turns]
    window.embeddings = await asyncio.gather(*embedding_tasks)

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(window.embeddings) - 1):
        for j in range(i + 1, len(window.embeddings)):
            sim = cosine_similarity([window.embeddings[i]], [window.embeddings[j]])[0][
                0
            ]
            similarities.append(sim)

    # Calculate window metrics
    window.window_similarity = np.mean(similarities)
    # Convert similarity to drift score (1 - similarity, normalized to [0, 1])
    window.drift_score = 1 - window.window_similarity

    return window


def prepare_windows(
    conversation_data: ConversationData,
    window_size: int = 3,
) -> List[TurnWindow]:
    """Prepare sliding windows from conversation data.

    Args:
        conversation_data: ConversationData object
        window_size: Size of each sliding window

    Returns:
        List of TurnWindow objects
    """
    windows = []
    for conv in conversation_data.conversations:
        turns = conv["turns"]
        if len(turns) >= window_size:
            # Create sliding windows
            for i in range(len(turns) - window_size + 1):
                window_turns = turns[i : i + window_size]
                windows.append(TurnWindow(turns=window_turns))
    return windows


async def prepare_training_data_async(
    conversation_data: ConversationData,
    window_size: int = 8,
    batch_size: int = 64,
    max_workers: int = 8,
    use_cache: bool = True,
    force_recompute: bool = False,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    """Prepare training data asynchronously using sliding windows.

    Args:
        conversation_data: ConversationData object containing conversations
        window_size: Size of the sliding window (default: 8)
        batch_size: Number of windows to process in parallel
        max_workers: Maximum number of worker threads
        use_cache: Whether to use cached data
        force_recompute: Whether to force recomputation even if cache exists
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        DataSplit object containing train/val/test tensors
    """
    # Check cache first
    if use_cache and not force_recompute:
        cache_key = get_cache_key(conversation_data, window_size)
        cached_data = load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

    # Initialize Ollama client
    ollama = OllamaWrapper(embedding_model="bge-m3")

    # Prepare all windows
    windows = prepare_windows(conversation_data, window_size)
    print(f"Created {len(windows)} windows of size {window_size}")

    # Process in batches
    processed_windows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(windows), batch_size), desc="Processing batches"):
            batch = windows[i : i + batch_size]
            processed_batch = await asyncio.gather(
                *[process_window(window, ollama, executor) for window in batch]
            )
            processed_windows.extend(processed_batch)

    # Convert to tensors
    window_embeddings = torch.tensor(
        np.array([np.concatenate(window.embeddings) for window in processed_windows]),
        dtype=torch.float32,
    )
    drift_scores = torch.tensor(
        np.array([window.drift_score for window in processed_windows]),
        dtype=torch.float32,
    )

    # Split data
    data_split = split_data(
        window_embeddings,
        drift_scores,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    # Save to cache if enabled
    if use_cache:
        cache_key = get_cache_key(conversation_data, window_size)
        print(f"Saving prepared data to cache: {get_cache_path() / f'{cache_key}.npz'}")
        save_to_cache(data_split, cache_key)

    return data_split


def prepare_training_data(
    conversation_data: ConversationData,
    window_size: int = 8,
    batch_size: int = 16,
    max_workers: int = 4,
    use_cache: bool = True,
    force_recompute: bool = False,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    """Synchronous wrapper for async data preparation.

    Args:
        conversation_data: ConversationData object containing conversations
        window_size: Size of the sliding window (default: 8)
        batch_size: Number of windows to process in parallel
        max_workers: Maximum number of worker threads
        use_cache: Whether to use cached data
        force_recompute: Whether to force recomputation even if cache exists
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        DataSplit object containing train/val/test tensors
    """
    return asyncio.run(
        prepare_training_data_async(
            conversation_data,
            window_size,
            batch_size,
            max_workers,
            use_cache,
            force_recompute,
            val_size,
            test_size,
            random_state,
        )
    )
