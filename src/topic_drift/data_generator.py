import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi, Repository
import tempfile
import os
from dotenv import load_dotenv
from uuid import uuid4
from tqdm.auto import tqdm
from topic_drift.data_types import ConversationData
from topic_drift.llm_wrapper import OllamaWrapper

# Load environment variables from .env file
load_dotenv()

NUM_CONVERSATIONS = 0


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


def generate_synthetic_data(
    num_conversations: int = NUM_CONVERSATIONS,
    chat_model: str = "qwen2.5-coder:32b",
    save_path: Optional[str] = "data",
    hf_repo: Optional[str] = "leonvanbokhorst/topic-drift",
    save_interval: int = 50,
) -> ConversationData:
    """Generate synthetic conversation data without drift labels."""
    print("Initializing Ollama client...")
    ollama = OllamaWrapper(chat_model=chat_model)

    conversations = []
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
