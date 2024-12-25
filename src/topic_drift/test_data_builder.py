"""Test script for ConversationalDatasetBuilder.

This script tests the dataset builder with real data
and validates the dataset balance for topic drift prediction.
"""

from pathlib import Path
from typing import Dict, List
import sys
import traceback
from datasets import Dataset, DatasetDict, load_dataset
try:
    from data_builder import ConversationalDatasetBuilder, Conversation
except ImportError as e:
    print("Error importing data_builder. Make sure you're running from the correct directory.")
    print(f"Error details: {str(e)}")
    sys.exit(1)
    
import numpy as np
from collections import defaultdict

def print_dataset_info(dataset, name: str):
    """Print dataset structure and first example."""
    try:
        print(f"\n=== {name} Dataset Info ===")
        print("Features:", dataset.features)
        print("\nFirst example:")
        print(dataset[0])
        print("=" * 50)
    except Exception as e:
        print(f"Error printing dataset info: {str(e)}")

def analyze_task_suitability(dataset: Dict[str, List[Conversation]]):
    """Analyze dataset suitability for topic drift prediction."""
    print("\n=== Task Suitability Analysis ===")
    
    # Collect statistics
    stats = defaultdict(lambda: defaultdict(int))
    total_turns = 0
    total_transitions = 0
    total_convs = 0
    
    for source, conversations in dataset.items():
        try:
            print(f"\n{source} Analysis:")
            if not conversations:
                print("No conversations available")
                continue
                
            source_turns = 0
            source_transitions = 0
            
            # Analyze turn pairs
            turn_pairs = []
            for conv in conversations:
                total_convs += 1
                source_turns += len(conv.turns)
                source_transitions += len(conv.transition_points)
                
                # Collect turn pairs for transition analysis
                for i in range(len(conv.turns) - 1):
                    is_transition = i in conv.transition_points
                    turn_pair = {
                        'text1': conv.turns[i].text,
                        'text2': conv.turns[i+1].text,
                        'is_transition': is_transition,
                        'transition_type': conv.transition_types[conv.transition_points.index(i)] if is_transition else "none",
                        'length1': len(conv.turns[i].text.split()),
                        'length2': len(conv.turns[i+1].text.split()),
                        'has_markers': bool(conv.turns[i+1].topic_markers)
                    }
                    turn_pairs.append(turn_pair)
            
            total_turns += source_turns
            total_transitions += source_transitions
            
            # Compute statistics
            avg_turns = source_turns / len(conversations)
            transition_rate = source_transitions / source_turns if source_turns > 0 else 0
            
            # Analyze turn pairs
            marked_transitions = sum(1 for p in turn_pairs if p['is_transition'] and p['has_markers'])
            unmarked_transitions = sum(1 for p in turn_pairs if p['is_transition'] and not p['has_markers'])
            
            print(f"Conversations: {len(conversations)}")
            print(f"Average turns per conversation: {avg_turns:.1f}")
            print(f"Topic transition rate: {transition_rate:.2%}")
            print(f"Marked vs unmarked transitions: {marked_transitions}/{unmarked_transitions}")
            
            # Length distribution of turns around transitions
            transition_lengths = [(p['length1'], p['length2']) for p in turn_pairs if p['is_transition']]
            if transition_lengths:
                avg_pre_trans = np.mean([l[0] for l in transition_lengths])
                avg_post_trans = np.mean([l[1] for l in transition_lengths])
                print(f"Average words before transition: {avg_pre_trans:.1f}")
                print(f"Average words after transition: {avg_post_trans:.1f}")
        except Exception as e:
            print(f"Error analyzing {source}: {str(e)}")
            traceback.print_exc()
    
    if total_convs > 0:
        print("\nOverall Statistics:")
        print(f"Total conversations: {total_convs}")
        print(f"Total turns: {total_turns}")
        print(f"Total transitions: {total_transitions}")
        print(f"Overall transition rate: {total_transitions/total_turns:.2%}")
    else:
        print("\nNo conversations available for analysis")

def main():
    print("Starting dataset builder test...")
    
    try:
        # Initialize builder with cache directory
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        builder = ConversationalDatasetBuilder(cache_dir)
        print("Initialized ConversationalDatasetBuilder")
        
        # Test languages (focusing on English)
        languages = ["en"]
        print(f"\nTesting with languages: {languages}")
        
        # Check if we have cached data
        cache_info = builder.get_dataset_info()
        print("\nCache status:", cache_info.get("status", "Found cached dataset"))
        
        # Build dataset with 500 examples from DailyDialog
        print("\nBuilding dataset with 500 examples from DailyDialog...")
        dataset = builder.build_dataset(
            languages,
            use_cache=False,  # Don't use cache to ensure fresh data
            balance=True,
            max_examples={
                'daily_dialog': 500,     # 500 examples from DailyDialog
            }
        )
        print("Dataset built successfully")
        
        # Print detailed statistics
        print("\nDetailed Dataset Analysis:")
        for source, conversations in dataset.items():
            try:
                print(f"\n=== {source} Analysis ===")
                if not conversations:
                    print("No conversations available")
                    continue
                
                # Basic stats
                print("\nBasic Statistics:")
                print(f"Total conversations: {len(conversations)}")
                avg_turns = sum(len(conv.turns) for conv in conversations) / len(conversations)
                print(f"Average turns per conversation: {avg_turns:.1f}")
                
                # Language distribution
                print("\nLanguage Distribution:")
                lang_dist = defaultdict(int)
                for conv in conversations:
                    lang_dist[conv.language] += 1
                for lang, count in lang_dist.items():
                    print(f"{lang}: {count/len(conversations):.1%}")
                
                # Turn analysis
                print("\nTurn Analysis:")
                all_turns = [turn for conv in conversations for turn in conv.turns]
                turn_lengths = [len(turn.text.split()) for turn in all_turns]
                print(f"Total turns: {len(all_turns)}")
                print(f"Average words per turn: {np.mean(turn_lengths):.1f}")
                print(f"Word length distribution:")
                print(f"  Min: {min(turn_lengths)}")
                print(f"  25th percentile: {np.percentile(turn_lengths, 25):.1f}")
                print(f"  Median: {np.median(turn_lengths):.1f}")
                print(f"  75th percentile: {np.percentile(turn_lengths, 75):.1f}")
                print(f"  Max: {max(turn_lengths)}")
                
                # Topic transitions
                print("\nTopic Transition Analysis:")
                total_transitions = sum(len(conv.transition_points) for conv in conversations)
                print(f"Total transitions: {total_transitions}")
                print(f"Average transitions per conversation: {total_transitions/len(conversations):.1f}")
                
                # Transition types
                trans_types = [t for conv in conversations for t in conv.transition_types]
                if trans_types:
                    print("\nTransition Type Distribution:")
                    for t_type in set(trans_types):
                        print(f"{t_type}: {trans_types.count(t_type)/len(trans_types):.1%}")
                
                # Topic markers
                print("\nTopic Marker Analysis:")
                all_markers = [
                    marker
                    for conv in conversations
                    for turn in conv.turns
                    for marker in (turn.topic_markers or [])
                ]
                if all_markers:
                    print("Most common topic markers:")
                    marker_counts = defaultdict(int)
                    for marker in all_markers:
                        marker_counts[marker] += 1
                    for marker, count in sorted(marker_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"  {marker}: {count}")
                
                # Quality distribution
                print("\nQuality Score Distribution:")
                quality_scores = [conv.quality_score for conv in conversations if conv.quality_score is not None]
                if quality_scores:
                    print(f"  Min: {min(quality_scores):.3f}")
                    print(f"  Average: {np.mean(quality_scores):.3f}")
                    print(f"  Max: {max(quality_scores):.3f}")
                
                # Sample high-quality conversation
                print("\nSample High-Quality Conversation:")
                best_conv = max(conversations, key=lambda x: x.quality_score or 0)
                print(f"Quality score: {best_conv.quality_score:.3f}")
                print(f"Number of turns: {len(best_conv.turns)}")
                print(f"Number of transitions: {len(best_conv.transition_points)}")
                print("\nFirst 3 turns:")
                for i, turn in enumerate(best_conv.turns[:3]):
                    print(f"[Turn {i+1}] {turn.text}")
                if best_conv.transition_points:
                    first_trans = best_conv.transition_points[0]
                    print(f"\nFirst transition (turns {first_trans} -> {first_trans+1}):")
                    print(f"Before: {best_conv.turns[first_trans].text}")
                    print(f"After:  {best_conv.turns[first_trans+1].text}")
                    print(f"Type:   {best_conv.transition_types[0]}")
                
            except Exception as e:
                print(f"Error analyzing {source}: {str(e)}")
                traceback.print_exc()
        
        # Overall dataset analysis
        print("\n=== Overall Dataset Analysis ===")
        total_convs = sum(len(convs) for convs in dataset.values())
        total_turns = sum(len(conv.turns) for convs in dataset.values() for conv in convs)
        total_transitions = sum(len(conv.transition_points) for convs in dataset.values() for conv in convs)
        
        print(f"Total conversations: {total_convs}")
        print(f"Total turns: {total_turns}")
        print(f"Total transitions: {total_transitions}")
        print(f"Overall transition rate: {total_transitions/total_turns:.2%}")
        
        # Language balance
        all_langs = set(conv.language for convs in dataset.values() for conv in convs)
        print("\nOverall Language Distribution:")
        for lang in all_langs:
            count = sum(1 for convs in dataset.values() for conv in convs if conv.language == lang)
            print(f"{lang}: {count/total_convs:.1%}")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
        
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 