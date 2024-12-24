"""Module for cleaning and deduplicating conversation data."""

import json
from pathlib import Path
import tempfile
from typing import Set, Dict, List, Tuple
from huggingface_hub import HfApi
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
import os
import time

from topic_drift.data_loader import load_from_huggingface
from topic_drift.data_types import ConversationData


def analyze_conversations(conversations: List[Dict]) -> Dict:
    """Analyze conversation statistics.

    Args:
        conversations: List of conversation dictionaries

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_conversations": len(conversations),
        "total_turns": sum(len(conv["turns"]) for conv in conversations),
        "avg_turns_per_conv": np.mean([len(conv["turns"]) for conv in conversations]),
        "turn_lengths": [len(turn) for conv in conversations for turn in conv["turns"]],
        "unique_speakers": set(),
    }

    # Analyze turn patterns
    turn_patterns = defaultdict(int)
    for conv in conversations:
        pattern = len(conv["turns"])
        turn_patterns[pattern] += 1

    stats["turn_patterns"] = dict(turn_patterns)
    stats["avg_turn_length"] = np.mean(stats["turn_lengths"])
    stats["max_turn_length"] = max(stats["turn_lengths"])
    stats["min_turn_length"] = min(stats["turn_lengths"])

    return stats


def print_stats_comparison(before: Dict, after: Dict) -> None:
    """Print comparison of before and after statistics.

    Args:
        before: Statistics before cleaning
        after: Statistics after cleaning
    """
    print("\n=== Cleaning Results ===")
    print(
        f"Total conversations: {before['total_conversations']} -> {after['total_conversations']}"
    )
    print(
        f"Removed duplicates: {before['total_conversations'] - after['total_conversations']}"
    )
    print(
        f"Duplicate ratio: {((before['total_conversations'] - after['total_conversations']) / before['total_conversations']):.2%}"
    )

    print("\n=== Conversation Patterns ===")
    print("Before cleaning:")
    for turns, count in sorted(before["turn_patterns"].items()):
        print(
            f"  {turns} turns: {count} conversations ({count/before['total_conversations']:.2%})"
        )

    print("\nAfter cleaning:")
    for turns, count in sorted(after["turn_patterns"].items()):
        print(
            f"  {turns} turns: {count} conversations ({count/after['total_conversations']:.2%})"
        )

    print("\n=== Turn Statistics ===")
    print(
        f"Average turns per conversation: {before['avg_turns_per_conv']:.2f} -> {after['avg_turns_per_conv']:.2f}"
    )
    print(
        f"Average turn length: {before['avg_turn_length']:.2f} -> {after['avg_turn_length']:.2f}"
    )
    print(
        f"Turn length range: {before['min_turn_length']}-{before['max_turn_length']} -> {after['min_turn_length']}-{after['max_turn_length']}"
    )


def deduplicate_conversations(
    data: ConversationData,
) -> Tuple[ConversationData, Dict, Dict]:
    """Remove duplicate conversations based on their content and IDs.

    Args:
        data: Original conversation data

    Returns:
        Tuple of (cleaned data, before stats, after stats)
    """
    print("\nAnalyzing original conversations...")
    before_stats = analyze_conversations(data.conversations)

    seen_turns: Set[str] = set()
    seen_ids: Set[str] = set()
    unique_conversations = []
    duplicates = defaultdict(lambda: {"content": 0, "id": 0})

    print("\nDeduplicating conversations...")
    for conv in tqdm(data.conversations, desc="Processing conversations"):
        # Check for ID duplicates
        conv_id = conv.get("conversation_id", "")
        is_id_duplicate = conv_id in seen_ids if conv_id else False

        # Check for content duplicates
        turns_str = json.dumps(conv["turns"], sort_keys=True)
        is_content_duplicate = turns_str in seen_turns

        if not is_id_duplicate and not is_content_duplicate:
            # Completely unique conversation
            if conv_id:
                seen_ids.add(conv_id)
            seen_turns.add(turns_str)
            unique_conversations.append(conv)
        else:
            # Track duplicate type
            num_turns = len(conv["turns"])
            if is_id_duplicate:
                duplicates[num_turns]["id"] += 1
            if is_content_duplicate:
                duplicates[num_turns]["content"] += 1

    cleaned_data = ConversationData(conversations=unique_conversations)
    after_stats = analyze_conversations(cleaned_data.conversations)

    print("\n=== Duplicate Analysis ===")
    print("Duplicates by conversation length:")
    for turns, counts in sorted(duplicates.items()):
        print(f"  {turns} turns:")
        print(f"    - ID duplicates: {counts['id']}")
        print(f"    - Content duplicates: {counts['content']}")
        total = counts["id"] + counts["content"]
        if total > 0:
            print(f"    - Total duplicates: {total}")

    return cleaned_data, before_stats, after_stats


def save_to_huggingface(
    data: ConversationData,
    repo_id: str = "leonvanbokhorst/topic-drift",
    token: str = None,
) -> None:
    """Save cleaned data back to Hugging Face using HTTP API.

    Args:
        data: Cleaned conversation data
        repo_id: Hugging Face repository ID
        token: Hugging Face API token
    """
    print("\nSaving to Hugging Face...")

    # Get token from env if not provided
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "Hugging Face token not found. Set HF_TOKEN in .env or pass token parameter"
        )

    api = HfApi(token=token)

    # Create temp directory for file operations
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        jsonl_path = tmp_dir / "conversations.jsonl"

        # Save conversations to JSONL with progress bar
        print("Writing conversations to JSONL...")
        start_time = time.time()
        with open(jsonl_path, "w") as f:
            for conv in tqdm(data.conversations, desc="Writing conversations"):
                json.dump(conv, f)
                f.write("\n")
        write_time = time.time() - start_time
        print(f"Writing completed in {write_time:.2f} seconds")

        # Upload to Hugging Face
        print("Uploading to Hugging Face...")
        start_time = time.time()
        api.upload_file(
            path_or_fileobj=str(jsonl_path),
            path_in_repo="conversations.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update deduplicated dataset",
        )
        upload_time = time.time() - start_time
        print(f"Upload completed in {upload_time:.2f} seconds")


def main():
    """Load data, deduplicate, and save back to Hugging Face."""
    # Load raw data
    print("Loading data from Hugging Face...")
    data = load_from_huggingface(force_reload=True)  # Force reload to avoid cache
    print(f"Loaded {len(data.conversations)} conversations")

    # Deduplicate and get statistics
    cleaned_data, before_stats, after_stats = deduplicate_conversations(data)

    # Print detailed comparison
    print_stats_comparison(before_stats, after_stats)

    # Save back to Hugging Face
    save_to_huggingface(cleaned_data)
    print("\nDone! Dataset has been cleaned and uploaded.")


if __name__ == "__main__":
    main()
