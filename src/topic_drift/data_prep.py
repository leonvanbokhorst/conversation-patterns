"""Module for preparing conversation data for training."""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
import json

from topic_drift.data_types import ConversationData
from topic_drift.llm_wrapper import OllamaWrapper


@dataclass
class TurnPair:
    """Container for a pair of turns and their embeddings."""

    turn1: str
    turn2: str
    emb1: np.ndarray = None
    emb2: np.ndarray = None
    drift_label: float = None


def get_cache_path() -> Path:
    """Get the path to the cache directory."""
    cache_dir = Path.home() / ".cache" / "topic_drift" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(
    conversation_data: ConversationData, similarity_threshold: float
) -> str:
    """Generate a cache key for the prepared data.

    Args:
        conversation_data: The conversation data
        similarity_threshold: The similarity threshold used

    Returns:
        A unique cache key based on the data content and parameters
    """
    # Create a string representation of the data and parameters
    data_str = json.dumps(
        [conv["turns"] for conv in conversation_data.conversations], sort_keys=True
    )
    param_str = f"{similarity_threshold}"

    # Generate hash
    return hashlib.md5(f"{data_str}{param_str}".encode()).hexdigest()


def save_to_cache(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    labels: torch.Tensor,
    cache_key: str,
) -> None:
    """Save prepared data to cache.

    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        labels: Labels tensor
        cache_key: Cache key for the data
    """
    cache_dir = get_cache_path()
    np.savez(
        cache_dir / f"{cache_key}.npz",
        embeddings1=embeddings1.numpy(),
        embeddings2=embeddings2.numpy(),
        labels=labels.numpy(),
    )


def load_from_cache(
    cache_key: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load prepared data from cache.

    Args:
        cache_key: Cache key for the data

    Returns:
        Tuple of tensors if cache exists, None otherwise
    """
    cache_path = get_cache_path() / f"{cache_key}.npz"
    if not cache_path.exists():
        return None

    print(f"Loading prepared data from cache: {cache_path}")
    data = np.load(cache_path)
    return (
        torch.from_numpy(data["embeddings1"]).float(),
        torch.from_numpy(data["embeddings2"]).float(),
        torch.from_numpy(data["labels"]).float(),
    )


async def process_batch(
    turn_pairs: List[TurnPair],
    ollama: OllamaWrapper,
    similarity_threshold: float,
    executor: ThreadPoolExecutor,
) -> List[TurnPair]:
    """Process a batch of turn pairs asynchronously.

    Args:
        turn_pairs: List of turn pairs to process
        ollama: OllamaWrapper instance
        similarity_threshold: Threshold for determining topic drift
        executor: ThreadPoolExecutor for running embeddings in parallel

    Returns:
        List of processed turn pairs with embeddings and labels
    """
    loop = asyncio.get_event_loop()

    # Get embeddings for all turns in parallel
    async def get_embedding(text: str) -> np.ndarray:
        return await loop.run_in_executor(executor, ollama.get_embeddings, text)

    # Process all turn1 embeddings
    emb1_tasks = [get_embedding(pair.turn1) for pair in turn_pairs]
    emb1_results = await asyncio.gather(*emb1_tasks)

    # Process all turn2 embeddings
    emb2_tasks = [get_embedding(pair.turn2) for pair in turn_pairs]
    emb2_results = await asyncio.gather(*emb2_tasks)

    # Calculate drift labels
    for pair, emb1, emb2 in zip(turn_pairs, emb1_results, emb2_results):
        pair.emb1 = emb1
        pair.emb2 = emb2
        sim_score = cosine_similarity([emb1], [emb2])[0][0]
        pair.drift_label = 1.0 if sim_score < similarity_threshold else 0.0

    return turn_pairs


def prepare_turn_pairs(conversation_data: ConversationData) -> List[TurnPair]:
    """Prepare turn pairs from conversation data.

    Args:
        conversation_data: ConversationData object

    Returns:
        List of TurnPair objects
    """
    turn_pairs = []
    for conv in conversation_data.conversations:
        turns = conv["turns"]
        for i in range(len(turns) - 1):
            turn_pairs.append(TurnPair(turn1=turns[i], turn2=turns[i + 1]))
    return turn_pairs


async def prepare_training_data_async(
    conversation_data: ConversationData,
    similarity_threshold: float = 0.7,
    batch_size: int = 64,
    max_workers: int = 8,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare training data asynchronously by generating embeddings and labels.

    Args:
        conversation_data: ConversationData object containing conversations
        similarity_threshold: Threshold for determining topic drift
        batch_size: Number of turn pairs to process in parallel
        max_workers: Maximum number of worker threads
        use_cache: Whether to use cached data
        force_recompute: Whether to force recomputation even if cache exists

    Returns:
        Tuple of (embeddings1, embeddings2, labels) as torch tensors
    """
    # Check cache first
    if use_cache and not force_recompute:
        cache_key = get_cache_key(conversation_data, similarity_threshold)
        cached_data = load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

    # Initialize Ollama client
    ollama = OllamaWrapper(embedding_model="bge-m3")

    # Prepare all turn pairs
    turn_pairs = prepare_turn_pairs(conversation_data)

    # Process in batches
    processed_pairs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(turn_pairs), batch_size), desc="Processing batches"):
            batch = turn_pairs[i : i + batch_size]
            processed_batch = await process_batch(
                batch, ollama, similarity_threshold, executor
            )
            processed_pairs.extend(processed_batch)

    # Convert to tensors
    embeddings1 = torch.tensor(
        np.array([pair.emb1 for pair in processed_pairs]), dtype=torch.float32
    )
    embeddings2 = torch.tensor(
        np.array([pair.emb2 for pair in processed_pairs]), dtype=torch.float32
    )
    labels = torch.tensor(
        np.array([pair.drift_label for pair in processed_pairs]), dtype=torch.float32
    )

    # Save to cache if enabled
    if use_cache:
        cache_key = get_cache_key(conversation_data, similarity_threshold)
        print(f"Saving prepared data to cache: {get_cache_path() / f'{cache_key}.npz'}")
        save_to_cache(embeddings1, embeddings2, labels, cache_key)

    return embeddings1, embeddings2, labels


def prepare_training_data(
    conversation_data: ConversationData,
    similarity_threshold: float = 0.7,
    batch_size: int = 32,
    max_workers: int = 8,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synchronous wrapper for async data preparation.

    Args:
        conversation_data: ConversationData object containing conversations
        similarity_threshold: Threshold for determining topic drift
        batch_size: Number of turn pairs to process in parallel
        max_workers: Maximum number of worker threads
        use_cache: Whether to use cached data
        force_recompute: Whether to force recomputation even if cache exists

    Returns:
        Tuple of (embeddings1, embeddings2, labels) as torch tensors
    """
    return asyncio.run(
        prepare_training_data_async(
            conversation_data,
            similarity_threshold,
            batch_size,
            max_workers,
            use_cache,
            force_recompute,
        )
    )
