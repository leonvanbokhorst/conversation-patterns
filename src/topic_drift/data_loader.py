from huggingface_hub import HfApi
import json
import tempfile
import os
from pathlib import Path
from tqdm.auto import tqdm
from topic_drift.data_types import ConversationData
import hashlib


def get_cache_path() -> Path:
    """Get the path to the cache directory."""
    cache_dir = Path.home() / ".cache" / "topic_drift"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_key(repo_id: str) -> str:
    """Generate a cache key for the dataset."""
    return hashlib.md5(repo_id.encode()).hexdigest()


def load_from_huggingface(
    repo_id: str = "leonvanbokhorst/topic-drift",
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

    Raises:
        ValueError: If no token is provided and none found in environment
        Exception: If download fails or no data found
    """
    # Get token from env if not provided
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face token not found. Set HF_TOKEN in .env or pass token parameter"
        )

    cache_path = get_cache_path() / f"{get_cache_key(repo_id)}.jsonl"

    # Check cache first
    if use_cache and not force_reload and cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        with open(cache_path, "r") as f:
            conversations = [json.loads(line) for line in f]
        return ConversationData(conversations=conversations)

    print(f"Downloading from Hugging Face: {repo_id}")
    api = HfApi(token=token)

    # Create temp directory for file operations
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            api.snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=tmp_dir,
                token=token,
                ignore_patterns=[".*"],
            )

            # Load conversations from JSONL
            conversations = []
            jsonl_path = Path(tmp_dir) / "conversations.jsonl"
            if not jsonl_path.exists():
                raise Exception(f"No data found in {repo_id}")

            print("Reading conversations from downloaded file...")
            with open(jsonl_path, "r") as f:
                for line in tqdm(f, desc="Loading conversations"):
                    conversations.append(json.loads(line))

            # Update cache if enabled
            if use_cache:
                print(f"Updating cache: {cache_path}")
                with open(cache_path, "w") as f:
                    for conv in conversations:
                        json.dump(conv, f)
                        f.write("\n")

            return ConversationData(conversations=conversations)

        except Exception as e:
            raise Exception(f"Failed to load data from {repo_id}: {str(e)}") from e
