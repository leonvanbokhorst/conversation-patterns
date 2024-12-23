import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ollama import Client
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi, Repository
import tempfile
import os
from dotenv import load_dotenv
from uuid import uuid4
from tqdm.auto import tqdm

# Load environment variables from .env file
load_dotenv()

NUM_CONVERSATIONS = 250


@dataclass
class ConversationData:
    """Container for raw conversation data."""

    conversations: List[Dict[str, any]]  # List of conversations with their turns


class OllamaWrapper:
    def __init__(self, chat_model: str = "llama3.2", embedding_model: str = "bge-m3"):
        """Initialize Ollama client with specified models."""
        self.client = Client()
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def generate_text(self, prompt: str) -> str:
        """Generate text using Ollama chat model."""
        response = self.client.chat(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7},
        )
        return response["message"]["content"]

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using Ollama embedding model."""
        response = self.client.embeddings(model=self.embedding_model, prompt=text)
        return np.array(response["embedding"])


def generate_conversation(ollama: OllamaWrapper) -> List[str]:
    """Generate a multi-turn conversation."""
    prompt = """Generate a natural 4-turn conversation between two people. 
    Make it flow naturally, with varied topics and responses. Start right away without greeting Speaker2.
    Format: Speaker1: [text]\nSpeaker2: [text]\n...
    
    Example:
    Speaker1: Uhm, I was thinking about going to the gym today. Want to join?
    Speaker2: Oh, I'm not sure, I'm not feeling well, so...
    Speaker1: I'm sorry to hear that. What's wrong?
    Speaker2: I have a headache since yesterday. It's been bothering me all day.
    Speaker1: Oh, that's not good. I hope you feel better soon.
    Speaker2: What can I do? I'll try to rest.
    Speaker1: Well, let me know if you need anything.
    Speaker2: I will, thanks.
    """

    response = ollama.generate_text(prompt)
    return [t.split(": ")[1] for t in response.split("\n") if ": " in t]


def save_conversation_data(data: ConversationData, base_path: str = "data") -> None:
    """Save raw conversation data in a flat, dataset-friendly format."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # Save dataset as JSONL with conversations and their turns
    with open(base_path / "conversations.jsonl", "w") as f:
        for record in data.conversations:
            json.dump(record, f)
            f.write("\n")


def load_conversation_data(base_path: str = "data") -> Optional[ConversationData]:
    """Load conversation data from flat format."""
    base_path = Path(base_path)
    if not (base_path / "conversations.jsonl").exists():
        return None

    conversations = []
    with open(base_path / "conversations.jsonl", "r") as f:
        conversations.extend(json.loads(line) for line in f)
    return ConversationData(conversations=conversations)


def sync_with_huggingface(
    data: Optional[ConversationData] = None,
    repo_id: str = "leonvanbokhorst/topic-drift",
    mode: str = "auto",
    token: Optional[str] = None,
) -> Optional[ConversationData]:
    """Sync conversation data with Hugging Face repository."""
    # Get token from env if not provided
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face token not found. Set HF_TOKEN in .env or pass token parameter"
        )

    api = HfApi(token=token)

    # Create repository if it doesn't exist
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception:
        print(f"Creating new dataset repository: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)

    # Create temp directory for file operations
    with tempfile.TemporaryDirectory() as tmp_dir:
        if mode in ["download", "auto"]:
            try:
                api.snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=tmp_dir,
                    token=token,
                    ignore_patterns=[".*"],
                )
                loaded_data = load_conversation_data(tmp_dir)
                if loaded_data is not None:
                    print(f"Successfully loaded data from {repo_id}")
                    return loaded_data
                elif mode == "download":
                    print(f"No data found in {repo_id}")
                    return None
            except Exception as e:
                if mode == "download":
                    raise Exception(
                        f"Failed to download from {repo_id}: {str(e)}"
                    ) from e
                print(f"No existing data found in {repo_id}, proceeding with upload")

        if mode in ["upload", "auto"] and data is not None:
            try:
                repo = Repository(
                    local_dir=tmp_dir,
                    clone_from=repo_id,
                    repo_type="dataset",
                    use_auth_token=token,
                )
                save_conversation_data(data, tmp_dir)
                repo.push_to_hub(
                    commit_message="Update topic drift dataset", blocking=True
                )
                print(f"Successfully uploaded data to {repo_id}")
            except Exception as e:
                raise Exception(f"Failed to upload to {repo_id}: {str(e)}") from e

    return None


def calculate_conversation_metrics(turns: List[str]) -> Dict[str, float]:
    """Calculate various metrics for a conversation."""
    # Basic length metrics
    turn_lengths = [len(turn.split()) for turn in turns]
    char_lengths = [len(turn) for turn in turns]

    return {
        "num_turns": len(turns),
        "avg_turn_length": sum(turn_lengths) / len(turns),
        "total_turn_length": sum(turn_lengths),
        "min_turn_length": min(turn_lengths),
        "max_turn_length": max(turn_lengths),
        "turn_length_variance": np.var(turn_lengths),
        "avg_chars_per_turn": sum(char_lengths) / len(turns),
        "avg_word_length": sum(char_lengths) / sum(turn_lengths),
        "question_ratio": (sum("?" in turn for turn in turns) / len(turns)),
        "exclamation_ratio": (sum("!" in turn for turn in turns) / len(turns)),
        "capitalization_ratio": sum(
            sum(bool(c.isupper()) for c in turn) / len(turn) for turn in turns
        )
        / len(turns),
    }


def generate_synthetic_data(
    num_conversations: int = NUM_CONVERSATIONS,
    chat_model: str = "hermes3",
    save_path: Optional[str] = "data",
    hf_repo: Optional[str] = "leonvanbokhorst/topic-drift",
    save_interval: int = 50,
) -> ConversationData:
    """Generate synthetic conversation data without drift labels."""
    # Try to load existing data
    existing_conversations = []
    if hf_repo:
        loaded_data = sync_with_huggingface(repo_id=hf_repo, mode="download")
        if loaded_data is not None:
            existing_conversations = loaded_data.conversations
            print(f"Loaded {len(existing_conversations)} existing conversations")
    elif save_path:
        loaded_data = load_conversation_data(save_path)
        if loaded_data is not None:
            existing_conversations = loaded_data.conversations
            print(f"Loaded {len(existing_conversations)} existing conversations")

    # If num_conversations is 0, just return existing data
    if num_conversations == 0:
        if not existing_conversations:
            print("No existing conversations found")
            return ConversationData(conversations=[])
        return ConversationData(conversations=existing_conversations)

    print("Initializing Ollama client...")
    ollama = OllamaWrapper(chat_model=chat_model)

    conversations = existing_conversations.copy()  # Start with existing conversations
    print(f"Generating {num_conversations} new conversations...")
    for i in tqdm(range(num_conversations)):
        conversation_id = str(uuid4())
        turns = generate_conversation(ollama)

        # Get all metrics
        metrics = calculate_conversation_metrics(turns)

        conversation_data = {
            "conversation_id": conversation_id,
            "turns": turns,
            **metrics,  # Unpack all metrics into the conversation data
        }
        conversations.append(conversation_data)

        # Periodic save to prevent data loss
        if (i + 1) % save_interval == 0:
            interim_data = ConversationData(conversations=conversations)
            if save_path:
                print(f"\nSaving interim data after {i + 1} new conversations...")
                save_conversation_data(interim_data, save_path)
            if hf_repo:
                print(f"Syncing to Hugging Face after {i + 1} new conversations...")
                sync_with_huggingface(interim_data, repo_id=hf_repo, mode="upload")

    # Create final ConversationData object
    data = ConversationData(conversations=conversations)

    # Final save
    if save_path:
        save_conversation_data(data, save_path)

    if hf_repo:
        sync_with_huggingface(data, repo_id=hf_repo, mode="upload")

    return data
