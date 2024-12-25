"""Test script for the ConversationalDatasetBuilder."""

from pathlib import Path
import os
from dotenv import load_dotenv
from topic_drift.data_builder import ConversationalDatasetBuilder
import numpy as np
from collections import defaultdict
from datasets import Dataset, DatasetDict
from typing import Dict, List, Any
from huggingface_hub import HfApi
import re

def clean_text(text: str) -> str:
    """Clean text by fixing spacing and punctuation issues."""
    # Fix apostrophes and quotes
    text = re.sub(r'\s*\'\s*([tsdm]|ll|ve|re)\b', r"'\1", text, flags=re.IGNORECASE)  # Fix contractions
    text = re.sub(r'\s*n\s*\'\s*t\b', r"n't", text, flags=re.IGNORECASE)  # Fix "n't"
    text = re.sub(r'\s*\'\s*', "'", text)  # Fix other apostrophes
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.?!:;])', r'\1', text)
    
    # Ensure single space after punctuation
    text = re.sub(r'([,.?!:;])\s*', r'\1 ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def convert_to_hf_format(conversations: List[Any]) -> List[Dict]:
    """Convert conversations to Hugging Face dataset format."""
    dataset_dicts = []
    
    for conv in conversations:
        # Extract turns and metadata
        turns = [clean_text(turn.text) for turn in conv.turns]
        speakers = [turn.speaker_id for turn in conv.turns]
        topic_markers = [turn.topic_markers or [] for turn in conv.turns]
        cultural_markers = [turn.cultural_markers or [] for turn in conv.turns]
        
        # Create dataset entry
        entry = {
            'conversation': turns,
            'speakers': speakers,
            'topics': conv.topics,
            'transition_points': conv.transition_points,
            'transition_types': conv.transition_types,
            'topic_markers': topic_markers,
            'cultural_markers': cultural_markers,
            'quality_score': float(conv.quality_score) if conv.quality_score else 0.0,
            'language': conv.language,
            'source': conv.source
        }
        dataset_dicts.append(entry)
    
    return dataset_dicts

def upload_to_huggingface(dataset_dicts: List[Dict], repo_id: str):
    """Upload dataset to Hugging Face."""
    # Create train/validation/test splits
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(dataset_dicts))
    
    train_idx = int(0.8 * len(indices))
    val_idx = int(0.9 * len(indices))
    
    train_data = [dataset_dicts[i] for i in indices[:train_idx]]
    val_data = [dataset_dicts[i] for i in indices[train_idx:val_idx]]
    test_data = [dataset_dicts[i] for i in indices[val_idx:]]
    
    print(f"Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    # Create HF datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Delete the repository if it exists
    api = HfApi()
    try:
        api.delete_repo(repo_id=repo_id, token=os.getenv('HF_TOKEN'))
        print(f"Deleted existing repository: {repo_id}")
    except Exception as e:
        print(f"Repository does not exist or could not be deleted: {e}")
    
    # Push to hub
    dataset_dict.push_to_hub(
        repo_id,
        token=os.getenv('HF_TOKEN'),
        private=False
    )

def analyze_conversation_details(conversation):
    """Analyze detailed statistics for a single conversation."""
    stats = {
        "num_turns": len(conversation.turns),
        "num_transitions": len(conversation.transition_points),
        "transition_rate": len(conversation.transition_points) / (len(conversation.turns) - 1) if len(conversation.turns) > 1 else 0,
        "avg_words_per_turn": np.mean([len(turn.text.split()) for turn in conversation.turns]),
        "topic_markers": defaultdict(int),
        "cultural_markers": defaultdict(int)
    }
    
    # Analyze markers
    for turn in conversation.turns:
        for marker in (turn.topic_markers or []):
            stats["topic_markers"][marker] += 1
        for marker in (turn.cultural_markers or []):
            stats["cultural_markers"][marker] += 1
            
    return stats

def print_dataset_analysis(dataset):
    """Print detailed analysis of the dataset."""
    print("\nDetailed Dataset Analysis:")
    
    for source, conversations in dataset.items():
        print(f"\n=== {source} Analysis ===")
        
        # Basic statistics
        total_turns = sum(len(conv.turns) for conv in conversations)
        total_transitions = sum(len(conv.transition_points) for conv in conversations)
        
        print("\nBasic Statistics:")
        print(f"Total conversations: {len(conversations)}")
        print(f"Average turns per conversation: {total_turns/len(conversations):.1f}")
        
        # Language distribution
        lang_dist = defaultdict(int)
        for conv in conversations:
            lang_dist[conv.language] += 1
        
        print("\nLanguage Distribution:")
        for lang, count in lang_dist.items():
            print(f"{lang}: {count/len(conversations):.1%}")
        
        # Turn analysis
        turn_lengths = [len(turn.text.split()) for conv in conversations for turn in conv.turns]
        
        print("\nTurn Analysis:")
        print(f"Total turns: {total_turns}")
        print(f"Average words per turn: {np.mean(turn_lengths):.1f}")
        print("Word length distribution:")
        print(f"  Min: {min(turn_lengths)}")
        print(f"  25th percentile: {np.percentile(turn_lengths, 25):.1f}")
        print(f"  Median: {np.percentile(turn_lengths, 50):.1f}")
        print(f"  75th percentile: {np.percentile(turn_lengths, 75):.1f}")
        print(f"  Max: {max(turn_lengths)}")
        
        # Topic transition analysis
        print("\nTopic Transition Analysis:")
        print(f"Total transitions: {total_transitions}")
        print(f"Average transitions per conversation: {total_transitions/len(conversations):.1f}")
        
        # Transition type distribution
        transition_types = defaultdict(int)
        for conv in conversations:
            for t_type in conv.transition_types:
                transition_types[t_type] += 1
                
        total_types = sum(transition_types.values())
        print("\nTransition Type Distribution:")
        for t_type, count in transition_types.items():
            print(f"{t_type}: {count/total_types:.1%}")
        
        # Topic marker analysis
        all_markers = defaultdict(int)
        for conv in conversations:
            for turn in conv.turns:
                for marker in (turn.topic_markers or []):
                    all_markers[marker] += 1
                    
        print("\nTopic Marker Analysis:")
        print("Most common topic markers:")
        for marker, count in sorted(all_markers.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {marker}: {count}")
        
        # Quality analysis
        quality_scores = [conv.quality_score for conv in conversations if conv.quality_score is not None]
        
        print("\nQuality Score Distribution:")
        print(f"  Min: {min(quality_scores):.3f}")
        print(f"  Average: {np.mean(quality_scores):.3f}")
        print(f"  Max: {max(quality_scores):.3f}")
        
        # Sample high-quality conversation
        best_conv = max(conversations, key=lambda x: x.quality_score or 0)
        print("\nSample High-Quality Conversation:")
        print(f"Quality score: {best_conv.quality_score:.3f}")
        print(f"Number of turns: {len(best_conv.turns)}")
        print(f"Number of transitions: {len(best_conv.transition_points)}")
        
        print("\nFirst 3 turns:")
        for i in range(min(3, len(best_conv.turns))):
            print(f"[Turn {i+1}] {best_conv.turns[i].text}")
            
        if best_conv.transition_points:
            print("\nFirst transition (turns {} -> {}):".format(
                best_conv.transition_points[0],
                best_conv.transition_points[0] + 1
            ))
            print("Before: ", best_conv.turns[best_conv.transition_points[0]].text)
            print("After:  ", best_conv.turns[best_conv.transition_points[0] + 1].text)
            print("Type:  ", best_conv.transition_types[0])
    
    # Overall dataset analysis
    print("\n=== Overall Dataset Analysis ===")
    total_convs = sum(len(convs) for convs in dataset.values())
    total_turns = sum(sum(len(conv.turns) for conv in convs) for convs in dataset.values())
    total_transitions = sum(sum(len(conv.transition_points) for conv in convs) for convs in dataset.values())
    
    print(f"Total conversations: {total_convs}")
    print(f"Total turns: {total_turns}")
    print(f"Total transitions: {total_transitions}")
    print(f"Overall transition rate: {total_transitions/total_turns:.2%}")
    
    # Overall language distribution
    overall_lang_dist = defaultdict(int)
    for convs in dataset.values():
        for conv in convs:
            overall_lang_dist[conv.language] += 1
            
    print("\nOverall Language Distribution:")
    for lang, count in overall_lang_dist.items():
        print(f"{lang}: {count/total_convs:.1%}")

def main():
    """Main test function."""
    print("Starting dataset builder test...")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize builder
    cache_dir = Path("cache")
    builder = ConversationalDatasetBuilder(cache_dir)
    print("Initialized ConversationalDatasetBuilder")
    
    # Test parameters
    languages = ["en"]
    print("\nTesting with languages:", languages)
    
    # Check cache status
    cache_info = builder.get_dataset_info()
    print("\nCache status:", cache_info.get("status", "Unknown"))
    
    # Build dataset with all available examples
    print("\nBuilding dataset with all available examples from DailyDialog...")
    dataset = builder.build_dataset(
        languages=languages,
        use_cache=False,  # Force rebuild to get more examples
        balance=True,
        max_examples={'daily_dialog': 11000}  # Process all available examples
    )
    
    if dataset:
        print("Dataset built successfully")
        print_dataset_analysis(dataset)
        
        # Convert and upload to Hugging Face
        print("\nPreparing dataset for Hugging Face...")
        dataset_dicts = convert_to_hf_format(dataset['daily_dialog'])
        
        print(f"\nUploading dataset to Hugging Face (leonvanbokhorst/topic-drift-v2)...")
        upload_to_huggingface(dataset_dicts, "leonvanbokhorst/topic-drift-v2")
        print("Dataset uploaded successfully!")
    else:
        print("Failed to build dataset")

if __name__ == "__main__":
    main() 