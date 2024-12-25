from huggingface_hub import HfApi
import json
import tempfile
import os
from pathlib import Path
from tqdm.auto import tqdm
from topic_drift.data_types import ConversationData
import hashlib
from datasets import load_dataset


def get_cache_path() -> Path:
    """Get the path to the cache directory."""
    cache_dir = Path.home() / ".cache" / "topic_drift"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(repo_id: str) -> str:
    """Generate a cache key for the dataset."""
    return hashlib.md5(repo_id.encode()).hexdigest()


def load_from_huggingface(
    repo_id: str = "leonvanbokhorst/topic-drift-v2",
    token: str = None,
    use_cache: bool = True,
    force_reload: bool = False,
) -> ConversationData:
    """Load conversation data from Hugging Face repository.

    Args:
        repo_id: The Hugging Face repository ID
        token: Hugging Face API token. If None, will try to get from HF_TOKEN env var
        use_cache: Whether to use local cache
        force_reload: Whether to force reload from HF even if cache exists

    Returns:
        ConversationData object containing the loaded conversations
    """
    print(f"Loading dataset from Hugging Face: {repo_id}")
    dataset = load_dataset(repo_id)
    
    if dataset is None:
        raise ValueError("Failed to load dataset from Hugging Face")
    
    # Convert dataset to conversation format
    conversations = []
    for split in ['train', 'validation', 'test']:
        for example in dataset[split]:
            conversation = {
                'turns': example['conversation'],
                'speakers': example['speakers'],
                'topic_markers': example['topic_markers'],
                'transition_points': example['transition_points'],
                'quality_score': example.get('quality_score', 1.0)
            }
            conversations.append(conversation)
    
    print(f"Loaded {len(conversations)} conversations")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    return ConversationData(conversations=conversations)
