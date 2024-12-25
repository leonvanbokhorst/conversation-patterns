"""Dataset builder for multilingual conversational data.

This module handles the collection, generation, and validation of conversational data
from both real-world sources and synthetic generation using LLMs.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from datasets import load_dataset
import torch
from tqdm.auto import tqdm
import re
import json
import datetime

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    text: str
    language: str
    speaker_id: str
    timestamp: Optional[float] = None
    topic_markers: Optional[List[str]] = None
    cultural_markers: Optional[List[str]] = None

@dataclass
class Conversation:
    """Represents a full conversation with metadata."""
    turns: List[ConversationTurn]
    language: str
    topics: List[str]
    transition_points: List[int]  # Indices where topic transitions occur
    transition_types: List[str]   # "smooth", "abrupt", etc.
    source: str  # "real" or "synthetic"
    quality_score: Optional[float] = None

class ConversationalDatasetBuilder:
    """Builder for multilingual conversational datasets."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
    def build_dataset(
        self,
        languages: List[str],
        use_cache: bool = True,
        balance: bool = True,
        max_examples: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[Conversation]]:
        """Build the complete dataset from all sources.
        
        Args:
            languages: List of language codes to include
            use_cache: Whether to use cached data if available
            balance: Whether to balance the dataset for better training
            max_examples: Dict mapping source names to max number of examples to process
            
        Returns:
            Dict mapping source names to lists of conversations
        """
        # Load or build raw dataset
        dataset = self._load_or_build_raw_dataset(languages, use_cache, max_examples)
        
        if balance:
            dataset = self._balance_dataset(dataset)
            
        return dataset
        
    def _load_or_build_raw_dataset(
        self,
        languages: List[str],
        use_cache: bool,
        max_examples: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[Conversation]]:
        """Load from cache or build raw dataset."""
        cache_path = self.cache_dir / "processed_conversations.pt"
        cache_meta_path = self.cache_dir / "dataset_metadata.json"
        
        # Check if cached data exists and is valid
        if use_cache and cache_path.exists() and cache_meta_path.exists():
            try:
                with open(cache_meta_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Check if cache is valid for these languages
                if set(metadata['languages']) == set(languages):
                    print("\nLoading cached dataset...")
                    return torch.load(cache_path)
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
        
        print("\nBuilding dataset from scratch...")
        
        # Process each source
        dataset = {}
        
        # DailyDialog - English conversational dataset
        print("\nProcessing DailyDialog...")
        try:
            daily_dialog = load_dataset("daily_dialog", trust_remote_code=True)
            if max_examples and 'daily_dialog' in max_examples:
                daily_dialog['train'] = daily_dialog['train'].select(range(max_examples['daily_dialog']))
            dataset['daily_dialog'] = self._process_daily_dialog(daily_dialog, languages)
            print(f"Successfully processed {len(dataset['daily_dialog'])} DailyDialog conversations")
        except Exception as e:
            print(f"Error loading DailyDialog dataset: {str(e)}")
            dataset['daily_dialog'] = []
        
        # Cache the processed dataset
        if use_cache:
            print("\nCaching processed dataset...")
            try:
                torch.save(dataset, cache_path)
                
                # Save metadata
                metadata = {
                    'languages': languages,
                    'num_conversations': {
                        source: len(convs) for source, convs in dataset.items()
                    },
                    'created_at': str(datetime.datetime.now()),
                    'sources': list(dataset.keys())
                }
                with open(cache_meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print("Dataset cached successfully")
            except Exception as e:
                print(f"Error caching dataset: {str(e)}")
        
        return dataset
        
    def _balance_dataset(self, dataset: Dict[str, List[Conversation]]) -> Dict[str, List[Conversation]]:
        """Balance the dataset for better training.
        
        Balancing criteria:
        1. Turn length distribution
        2. Transition type distribution
        3. Source distribution
        """
        print("\nBalancing dataset...")
        balanced = {}
        
        # 1. Analyze current distributions
        stats = self._analyze_dataset_stats(dataset)
        
        # 2. Determine target distributions
        target_stats = {
            'turn_lengths': {
                'short': 0.3,    # < 10 words
                'medium': 0.5,   # 10-30 words
                'long': 0.2      # > 30 words
            },
            'transition_types': {
                'smooth': 0.3,   # Marked transitions
                'semantic': 0.5, # Unmarked but clear transitions
                'none': 0.2      # No clear transition
            }
        }
        
        # 3. Balance each source
        total_target = 1000  # Target total conversations
        
        for source, conversations in dataset.items():
            # Sort by quality score
            sorted_convs = sorted(
                conversations,
                key=lambda x: x.quality_score or 0,
                reverse=True
            )
            
            # Select top conversations
            quota = int(total_target * 0.8)  # Take 80% of target
            selected = sorted_convs[:quota]
            
            balanced[source] = selected
            
            print(f"\nBalanced {source}:")
            print(f"- Total conversations: {len(selected)}")
        
        print("\nDataset balance achieved:")
        final_stats = self._analyze_dataset_stats(balanced)
        for metric, dist in final_stats.items():
            print(f"\n{metric}:")
            for key, value in dist.items():
                print(f"  {key}: {value:.1%}")
        
        return balanced
        
    def _analyze_dataset_stats(self, dataset: Dict[str, List[Conversation]]) -> Dict:
        """Analyze current dataset statistics."""
        stats = {
            'languages': {},
            'turn_lengths': {
                'short': 0,
                'medium': 0,
                'long': 0
            },
            'transition_types': {
                'smooth': 0,
                'semantic': 0,
                'none': 0
            }
        }
        
        total_convs = sum(len(convs) for convs in dataset.values())
        total_turns = 0
        total_transitions = 0
        
        for conversations in dataset.values():
            for conv in conversations:
                # Language stats
                if conv.language not in stats['languages']:
                    stats['languages'][conv.language] = 0
                stats['languages'][conv.language] += 1
                
                # Turn length stats
                for turn in conv.turns:
                    total_turns += 1
                    words = len(turn.text.split())
                    if words < 10:
                        stats['turn_lengths']['short'] += 1
                    elif words < 30:
                        stats['turn_lengths']['medium'] += 1
                    else:
                        stats['turn_lengths']['long'] += 1
                
                # Transition stats
                total_transitions += len(conv.transition_points)
                for t_type in conv.transition_types:
                    stats['transition_types'][t_type] += 1
                stats['transition_types']['none'] += (
                    len(conv.turns) - 1 - len(conv.transition_points)
                )
        
        # Normalize stats
        for lang in stats['languages']:
            stats['languages'][lang] /= total_convs
            
        for length in stats['turn_lengths']:
            stats['turn_lengths'][length] /= total_turns
            
        total_turn_pairs = sum(len(conv.turns) - 1 for convs in dataset.values() for conv in convs)
        for t_type in stats['transition_types']:
            stats['transition_types'][t_type] /= total_turn_pairs
        
        return stats
        
    def _balance_source(
        self,
        conversations: List[Conversation],
        target_stats: Dict,
        max_convs: int
    ) -> List[Conversation]:
        """Balance conversations from a single source."""
        # Sort conversations by quality score
        conversations = sorted(conversations, key=lambda x: x.quality_score or 0, reverse=True)
        
        # Initialize balanced list with highest quality conversations
        balanced = []
        current_stats = {}
        
        # Add conversations while maintaining target distributions
        for conv in conversations:
            if len(balanced) >= max_convs:
                break
                
            # Check if adding this conversation improves balance
            temp_stats = self._analyze_dataset_stats({'temp': balanced + [conv]})
            
            # Compute distance from target distribution
            dist_before = self._distribution_distance(current_stats, target_stats) if current_stats else float('inf')
            dist_after = self._distribution_distance(temp_stats, target_stats)
            
            if dist_after < dist_before:
                balanced.append(conv)
                current_stats = temp_stats
        
        return balanced
        
    def _distribution_distance(self, current: Dict, target: Dict) -> float:
        """Compute distance between current and target distributions."""
        distance = 0
        
        for metric in target:
            if metric not in current:
                distance += 1
                continue
                
            # Compute Jensen-Shannon divergence
            for key in target[metric]:
                curr_val = current[metric].get(key, 0)
                target_val = target[metric][key]
                distance += abs(curr_val - target_val)
        
        return distance

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the cached dataset if available."""
        cache_meta_path = self.cache_dir / "dataset_metadata.json"
        
        if not cache_meta_path.exists():
            return {"status": "No cached dataset found"}
            
        try:
            with open(cache_meta_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"status": f"Error reading cache metadata: {str(e)}"}
            
    def clear_cache(self):
        """Clear all cached data."""
        cache_path = self.cache_dir / "processed_conversations.pt"
        cache_meta_path = self.cache_dir / "dataset_metadata.json"
        
        if cache_path.exists():
            cache_path.unlink()
        if cache_meta_path.exists():
            cache_meta_path.unlink()
            
        print("Cache cleared successfully")

    def _process_daily_dialog(self, dataset, languages: List[str]) -> List[Conversation]:
        """Process Daily Dialog dataset.
        
        The DailyDialog dataset contains:
        - dialog: List of conversation turns
        - act: Dialogue act labels for each turn
            1: inform, 2: question, 3: directive, 4: commissive
        - emotion: Emotion labels for each turn
            0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise
        """
        conversations = []
        
        # Only process if English is in requested languages (DailyDialog is English-only)
        if "en" not in languages:
            return []
            
        print("Processing DailyDialog dataset...")
        
        # Process each dialogue in the training split
        for idx, dialogue in enumerate(tqdm(dataset['train'], desc="Processing dialogues")):
            utterances = dialogue['dialog']
            acts = dialogue['act']
            emotions = dialogue['emotion']
            
            # Skip if too short
            if len(utterances) < 3:
                continue
                
            # Create conversation turns
            turns = []
            transition_points = []
            transition_types = []
            topics = ["General Conversation"]  # Default topic
            
            for i, (utt, act, emotion) in enumerate(zip(utterances, acts, emotions)):
                # Detect topic transitions based on dialogue acts and emotions
                if i > 0:
                    prev_act = acts[i-1]
                    prev_emotion = emotions[i-1]
                    
                    # Topic transition heuristics:
                    # - Act sequence: inform/question after directive/commissive
                    # - Emotion changes
                    # - Presence of topic markers
                    # - Semantic shifts
                    topic_markers = self._extract_topic_markers(utt)
                    has_act_transition = (act in [1, 2] and prev_act in [3, 4])
                    has_emotion_change = (emotion != prev_emotion and emotion != 0)
                    has_semantic_shift = self._detect_semantic_shift(turns[-1].text, utt)
                    
                    if topic_markers or has_act_transition or has_emotion_change or has_semantic_shift:
                        transition_points.append(i)
                        transition_types.append("smooth" if topic_markers else "semantic")
                        
                        # Try to extract subtopic
                        subtopic = self._extract_subtopic(utt, topics[-1])
                        if subtopic:
                            topics.append(subtopic)
                
                # Create turn with metadata
                turn = ConversationTurn(
                    text=utt,
                    language="en",
                    speaker_id=str(i % 2),  # Alternate between speakers
                    topic_markers=self._extract_topic_markers(utt),
                    cultural_markers=self._extract_cultural_markers(utt, "en")
                )
                turns.append(turn)
            
            # Create conversation object
            conversation = Conversation(
                turns=turns,
                language="en",
                topics=topics,
                transition_points=transition_points,
                transition_types=transition_types,
                source="real",
                quality_score=self._compute_quality_score(turns, transition_points)
            )
            
            conversations.append(conversation)
            
            # Progress update
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} dialogues...")
        
        print(f"Processed {len(conversations)} valid conversations from DailyDialog")
        return conversations
        
    def _extract_topic_markers(self, text: str) -> List[str]:
        """Extract topic transition markers from text.
        
        Uses multiple types of markers:
        1. Explicit topic changes (strong)
        2. Topic shifts and pivots (moderate)
        3. Subtle transitions (weak)
        4. Contextual transitions
        5. Discourse markers
        """
        markers = []
        text_lower = text.lower()
        
        # 1. Explicit Topic Changes (Strong)
        strong_markers = {
            # Direct topic changes
            "let's talk about", "let's discuss", "speaking of",
            "on the topic of", "regarding the matter of",
            "turning to", "moving on to", "switching to",
            
            # Topic boundaries
            "changing the subject",
            "on another topic",
            "on a different note",
            "that reminds me of something else",
            
            # Topic introductions
            "i want to bring up",
            "i'd like to discuss",
            "we need to talk about",
            "let me tell you about",
        }
        
        # 2. Topic Shifts and Pivots (Moderate)
        moderate_markers = {
            # Natural transitions
            "by the way",
            "speaking of which",
            "that reminds me",
            "now that you mention it",
            
            # Topic development
            "on a related note",
            "in the same vein",
            "similarly",
            "along those lines",
            
            # Topic exploration
            "regarding",
            "concerning",
            "as for",
            "about that",
            
            # Topic continuation
            "while we're on the subject",
            "since we're talking about",
            "following up on that",
        }
        
        # 3. Subtle Transitions (Weak)
        weak_markers = {
            # Additive
            "also",
            "besides",
            "moreover",
            "furthermore",
            "in addition",
            
            # Comparative
            "similarly",
            "likewise",
            "in comparison",
            
            # Contrastive
            "however",
            "on the other hand",
            "in contrast",
            
            # Temporal
            "meanwhile",
            "at the same time",
            "in the meantime",
        }
        
        # 4. Contextual Transition Patterns
        contextual_patterns = [
            # Question-based transitions
            r"speaking of .+, what",
            r"that brings up .+, how",
            r"while we're discussing .+, could",
            
            # Reference-based transitions
            r"that makes me think of .+",
            r"this relates to .+",
            r"this is similar to .+",
            
            # Topic exploration
            r"if we consider .+",
            r"looking at .+ from another angle",
            r"thinking about .+ differently",
        ]
        
        # 5. Discourse Markers
        discourse_markers = {
            # Topic initiation
            "first", "firstly", "to begin with", "to start with",
            
            # Topic development
            "second", "secondly", "third", "thirdly",
            "next", "then", "subsequently",
            
            # Topic conclusion
            "finally", "lastly", "in conclusion",
            "to sum up", "in summary",
        }
        
        # Helper function for contextual analysis
        def get_marker_context(marker: str, text: str) -> Optional[str]:
            """Extract context around a marker."""
            try:
                start_idx = text.index(marker)
                # Get surrounding context (up to 3 words before and after)
                words = text.split()
                marker_words = marker.split()
                marker_word_count = len(marker_words)
                
                for i in range(len(words) - marker_word_count + 1):
                    if " ".join(words[i:i + marker_word_count]) == marker:
                        start = max(0, i - 3)
                        end = min(len(words), i + marker_word_count + 3)
                        return " ".join(words[start:end])
                return None
            except ValueError:
                return None
        
        # Check strong markers with context
        for marker in strong_markers:
            if marker in text_lower:
                context = get_marker_context(marker, text_lower)
                if context:
                    markers.append(f"strong:{marker}|{context}")
        
        # Check moderate markers with context
        for marker in moderate_markers:
            if marker in text_lower:
                context = get_marker_context(marker, text_lower)
                if context:
                    markers.append(f"moderate:{marker}|{context}")
        
        # Check weak markers with context
        for marker in weak_markers:
            if marker in text_lower:
                context = get_marker_context(marker, text_lower)
                if context:
                    markers.append(f"weak:{marker}|{context}")
        
        # Check contextual patterns
        for pattern in contextual_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                markers.append(f"contextual:{match.group()}")
        
        # Check discourse markers
        for marker in discourse_markers:
            if marker in text_lower:
                # Only count discourse markers at the start of sentences
                sentences = text_lower.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence.startswith(marker):
                        markers.append(f"discourse:{marker}")
        
        # Additional contextual analysis
        words = text_lower.split()
        
        # Check for topic word repetition
        if len(words) >= 2 and "topic" in words:
            topic_idx = words.index("topic")
            if topic_idx > 0:
                context = " ".join(words[max(0, topic_idx-2):min(len(words), topic_idx+3)])
                markers.append(f"topic_word:{context}")
        
        # Check for subject changes
        subject_markers = {"i", "you", "we", "they", "he", "she", "it"}
        for i, word in enumerate(words):
            if word in subject_markers and i > 0:
                if words[i-1] in {"but", "however", "although", "though"}:
                    context = " ".join(words[max(0, i-2):min(len(words), i+3)])
                    markers.append(f"subject_change:{context}")
        
        return markers
        
    def _extract_cultural_markers(self, text: str, language: str) -> List[str]:
        """Extract cultural-specific markers from text."""
        markers = []
        
        # English cultural markers
        if language == "en":
            cultural_phrases = [
                "you know",
                "well",
                "actually",
                "I mean",
                "sort of",
                "kind of",
                "like",
                "basically",
            ]
            
            text_lower = text.lower()
            for phrase in cultural_phrases:
                if phrase in text_lower:
                    markers.append(phrase)
        
        return markers
        
    def _compute_quality_score(
        self,
        turns: List[ConversationTurn],
        transition_points: List[int]
    ) -> float:
        """Compute a quality score for the conversation."""
        if not turns:
            return 0.0
            
        # Factors that contribute to quality
        factors = {
            "length": min(1.0, len(turns) / 10),  # Reward longer conversations up to 10 turns
            "transitions": min(1.0, len(transition_points) / 3),  # Reward transitions up to 3
            "avg_turn_length": 0.0,  # Will be computed
            "marker_presence": 0.0,  # Will be computed
        }
        
        # Compute average turn length score
        avg_words = np.mean([len(turn.text.split()) for turn in turns])
        factors["avg_turn_length"] = min(1.0, avg_words / 20)  # Reward up to 20 words per turn
        
        # Compute marker presence score
        total_markers = sum(
            len(turn.topic_markers or []) + len(turn.cultural_markers or [])
            for turn in turns
        )
        factors["marker_presence"] = min(1.0, total_markers / len(turns))
        
        # Weighted average of factors
        weights = {
            "length": 0.3,
            "transitions": 0.3,
            "avg_turn_length": 0.2,
            "marker_presence": 0.2
        }
        
        quality_score = sum(score * weights[factor] for factor, score in factors.items())
        return quality_score

    def _analyze_sentence_complexity(self, sentence: str) -> float:
        """Analyze sentence complexity based on length and structure.
        
        Returns:
            float: Complexity score between 0 and 1
        """
        # Basic complexity metrics
        words = sentence.split()
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Structural complexity indicators
        conjunctions = sum(1 for word in words if word.lower() in {
            'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'although'
        })
        
        # Punctuation complexity
        punctuation = sum(1 for char in sentence if char in {',', ';', ':', '(', ')'})
        
        # Compute complexity score (normalized to 0-1)
        length_score = min(1.0, word_count / 30)  # Cap at 30 words
        word_length_score = min(1.0, avg_word_length / 8)  # Cap at 8 chars
        structure_score = min(1.0, (conjunctions + punctuation) / 5)  # Cap at 5 markers
        
        return np.mean([length_score, word_length_score, structure_score])

    def _determine_optimal_window(
        self,
        sentences: List[str],
        source_type: str = "talk"
    ) -> int:
        """Determine optimal window size based on sentence analysis.
        
        Args:
            sentences: List of sentences to analyze
            source_type: Type of source ("talk" or "dialogue")
            
        Returns:
            int: Optimal number of sentences per turn
        """
        # Analyze sentence complexities
        complexities = [self._analyze_sentence_complexity(s) for s in sentences]
        avg_complexity = np.mean(complexities)
        
        if source_type == "talk":
            # For presentations/talks:
            # - Higher complexity -> smaller windows
            # - Lower complexity -> larger windows
            if avg_complexity > 0.7:
                return 2  # Complex sentences
            elif avg_complexity > 0.4:
                return 3  # Moderate complexity
            else:
                return 4  # Simple sentences
        else:
            # For natural dialogues:
            # - Usually keep as single utterances
            # - Maybe combine very simple consecutive utterances
            if avg_complexity < 0.3:
                return 2  # Combine simple utterances
            else:
                return 1  # Keep as single utterances

    def _group_sentences_into_turn(
        self,
        sentences: List[str],
        window_size: int,
        min_sentences: int = 1
    ) -> List[str]:
        """Group sentences into a turn while respecting semantic boundaries.
        
        Args:
            sentences: List of sentences to group
            window_size: Target number of sentences per turn
            min_sentences: Minimum sentences per turn
            
        Returns:
            List[str]: Grouped sentences forming a turn
        """
        if len(sentences) < min_sentences:
            return sentences
            
        # Try to find natural break points
        break_indicators = {
            "strong": [". However,", ". Nevertheless,", ". In contrast,", ". On the other hand,"],
            "moderate": [". Also,", ". Additionally,", ". Moreover,", ". Furthermore,"],
            "weak": [", and", ", but", ", so"]
        }
        
        # Start with target window size
        end_idx = min(window_size, len(sentences))
        combined_text = " ".join(sentences[:end_idx])
        
        # Check for break indicators to potentially extend or shrink the window
        for strength, indicators in break_indicators.items():
            for indicator in indicators:
                if indicator in combined_text:
                    # Found a natural break point
                    break_idx = combined_text.index(indicator) + 1
                    if strength == "strong":
                        # Strong break - respect it even if shorter
                        return sentences[:max(min_sentences, break_idx)]
                    elif strength == "moderate":
                        # Moderate break - use if close to target
                        if abs(break_idx - window_size) <= 1:
                            return sentences[:break_idx]
                    # Weak breaks only influence if exactly at target
        
        return sentences[:end_idx] 

    def _detect_semantic_shift(self, prev_text: str, curr_text: str) -> bool:
        """Detect semantic shift between two text segments.
        
        Uses multiple heuristics to detect topic transitions:
        1. Lexical signals (transition phrases)
        2. Semantic discontinuity (subject/object changes)
        3. Dialogue act shifts
        4. Question-answer patterns
        5. Reference shifts
        """
        # 1. Lexical signals
        shift_indicators = {
            # Strong shifts (high confidence)
            "strong": [
                "however", "nevertheless", "on the other hand", "in contrast",
                "turning to", "moving to", "switching to", "changing topics",
                "on another topic", "on a different note",
            ],
            # Topic development (medium confidence)
            "development": [
                "moreover", "furthermore", "additionally", "in addition",
                "speaking of", "regarding", "as for", "about",
                "another thing", "by the way",
            ],
            # Examples and elaboration (low confidence)
            "elaboration": [
                "for example", "for instance", "specifically",
                "to illustrate", "consider", "take", "let's say",
                "in particular", "especially",
            ],
            # Conclusions and summaries (high confidence)
            "conclusion": [
                "in conclusion", "finally", "to sum up", "overall",
                "in the end", "ultimately", "in summary", "therefore",
            ]
        }
        
        # Check for lexical signals
        curr_text_lower = curr_text.lower()
        prev_text_lower = prev_text.lower()
        
        # Strong shift indicators take precedence
        for indicator in shift_indicators["strong"]:
            if indicator in curr_text_lower:
                return True
        
        # Initialize shift score
        shift_score = 0
        
        # 2. Semantic discontinuity
        prev_subjects = self._extract_subjects(prev_text_lower)
        curr_subjects = self._extract_subjects(curr_text_lower)
        has_subject_overlap = bool(prev_subjects.intersection(curr_subjects))
        
        # 3. Dialogue act shifts
        prev_act = self._detect_dialogue_act(prev_text)
        curr_act = self._detect_dialogue_act(curr_text)
        has_act_shift = self._is_significant_act_shift(prev_act, curr_act)
        
        # 4. Question-answer patterns
        is_qa_pair = self._is_question_answer_pair(prev_text, curr_text)
        
        # 5. Reference shifts
        prev_refs = self._extract_references(prev_text_lower)
        curr_refs = self._extract_references(curr_text_lower)
        has_ref_continuity = bool(prev_refs.intersection(curr_refs))
        
        # Score calculation with adjusted weights
        
        # Major factors (max +3)
        if not has_subject_overlap and len(prev_subjects) > 0 and len(curr_subjects) > 0:
            shift_score += 3
        
        # Moderate factors (max +2)
        if has_act_shift and not is_qa_pair:
            shift_score += 2
            
        if any(marker in curr_text_lower for marker in shift_indicators["development"]):
            shift_score += 1
            
        if any(marker in curr_text_lower for marker in shift_indicators["conclusion"]):
            shift_score += 2
        
        # Minor factors (max +1)
        if not has_ref_continuity and len(prev_refs) > 0 and len(curr_refs) > 0:
            shift_score += 1
        
        # Continuity indicators (negative scores)
        if is_qa_pair:
            shift_score -= 2
            
        if any(marker in curr_text_lower for marker in shift_indicators["elaboration"]):
            shift_score -= 1
            
        if has_subject_overlap and has_ref_continuity:
            shift_score -= 1
        
        # Higher threshold for shift detection
        return shift_score >= 3  # Increased from 2 to 3

    def _extract_subjects(self, text: str) -> Set[str]:
        """Extract potential subjects from text."""
        # Common subject pronouns and determiners
        subject_indicators = {
            "i", "you", "he", "she", "it", "we", "they",
            "this", "that", "these", "those",
            "the", "a", "an", "my", "your", "his", "her", "its", "our", "their"
        }
        
        words = text.split()
        subjects = set()
        
        for i, word in enumerate(words):
            # Check for subject indicators
            if word.lower() in subject_indicators:
                # Include the following noun phrase
                phrase = []
                j = i
                while j < len(words) and not any(w in words[j] for w in ",.!?;:"):
                    phrase.append(words[j])
                    j += 1
                if phrase:
                    subjects.add(" ".join(phrase))
            
            # Check for capitalized words (potential proper nouns)
            elif word and word[0].isupper() and i > 0:
                subjects.add(word)
        
        return subjects

    def _detect_dialogue_act(self, text: str) -> str:
        """Detect the dialogue act of a text segment."""
        # Question detection
        if "?" in text or text.lower().startswith(("what", "who", "where", "when", "why", "how")):
            return "question"
            
        # Command/request detection
        if text.lower().startswith(("please", "could you", "would you", "let's")):
            return "directive"
            
        # Agreement/disagreement
        if any(word in text.lower() for word in ["yes", "no", "agree", "disagree", "okay", "fine"]):
            return "response"
            
        # Statement (default)
        return "statement"

    def _is_significant_act_shift(self, prev_act: str, curr_act: str) -> bool:
        """Determine if a dialogue act shift is significant for topic change."""
        # Define significant act transitions
        significant_shifts = {
            ("question", "statement"),  # New topic introduction after question
            ("response", "question"),   # New question after response
            ("directive", "statement"), # New topic after request
        }
        return (prev_act, curr_act) in significant_shifts

    def _is_question_answer_pair(self, prev_text: str, curr_text: str) -> bool:
        """Detect if two turns form a question-answer pair."""
        # Check if first turn is a question
        is_question = "?" in prev_text or prev_text.lower().startswith(
            ("what", "who", "where", "when", "why", "how")
        )
        
        if not is_question:
            return False
            
        # Check if second turn is an answer
        has_answer_indicator = any(
            curr_text.lower().startswith(word)
            for word in ["yes", "no", "well", "because", "i think", "maybe"]
        )
        
        return has_answer_indicator

    def _extract_references(self, text: str) -> Set[str]:
        """Extract referential expressions from text."""
        references = set()
        
        # Pronouns and demonstratives
        ref_words = {
            "it", "this", "that", "these", "those",
            "he", "she", "they", "him", "her", "them",
            "his", "hers", "their", "theirs"
        }
        
        # Add references from text
        words = text.split()
        for word in words:
            if word.lower() in ref_words:
                references.add(word.lower())
            
        return references

    def _extract_subtopic(self, text: str, main_topic: str) -> Optional[str]:
        """Extract potential subtopic from text segment."""
        # Keywords that often introduce subtopics
        subtopic_indicators = [
            "regarding",
            "concerning",
            "when it comes to",
            "in terms of",
            "speaking about",
            "focusing on"
        ]
        
        text_lower = text.lower()
        for indicator in subtopic_indicators:
            if indicator in text_lower:
                # Extract the phrase following the indicator
                start_idx = text_lower.index(indicator) + len(indicator)
                end_idx = text_lower.find(".", start_idx)
                if end_idx == -1:
                    end_idx = len(text_lower)
                
                subtopic = text[start_idx:end_idx].strip()
                # Only return if sufficiently different from main topic
                if subtopic and not self._is_similar_topic(subtopic, main_topic):
                    return subtopic
                
        return None
        
    def _is_similar_topic(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are semantically similar."""
        # Basic similarity check - could be improved with embeddings
        words1 = set(topic1.lower().split())
        words2 = set(topic2.lower().split())
        overlap = len(words1.intersection(words2))
        return overlap > min(len(words1), len(words2)) / 2 

    def _group_into_scenes(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences into scenes based on semantic similarity."""
        scenes = []
        current_scene = []
        
        for sentence in sentences:
            # Start new scene if:
            # 1. Current scene is empty
            # 2. Strong scene break detected
            # 3. Current scene is too long
            if (not current_scene or 
                self._is_scene_break(current_scene[-1], sentence) or 
                len(current_scene) >= 30):
                
                if current_scene:
                    scenes.append(current_scene)
                current_scene = [sentence]
            else:
                current_scene.append(sentence)
        
        # Add last scene
        if current_scene:
            scenes.append(current_scene)
        
        return scenes

    def _is_scene_break(self, prev_text: str, curr_text: str) -> bool:
        """Detect if there's a scene break between two texts."""
        # Scene break indicators
        break_indicators = [
            # Time jumps
            "later", "next day", "meanwhile", "after",
            # Location changes
            "at the", "in the", "outside", "inside",
            # Strong topic changes
            "suddenly", "elsewhere", "back to"
        ]
        
        # Check for break indicators
        curr_text_lower = curr_text.lower()
        if any(indicator in curr_text_lower for indicator in break_indicators):
            return True
        
        # Check for significant length difference (might indicate scene change)
        prev_words = len(prev_text.split())
        curr_words = len(curr_text.split())
        if abs(prev_words - curr_words) > 10:
            return True
        
        return False 

    def _process_daily_dialog(self, dataset, languages: List[str]) -> List[Conversation]:
        """Process Daily Dialog dataset.
        
        The DailyDialog dataset contains:
        - dialog: List of conversation turns
        - act: Dialogue act labels for each turn
            1: inform, 2: question, 3: directive, 4: commissive
        - emotion: Emotion labels for each turn
            0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise
        """
        conversations = []
        
        # Only process if English is in requested languages (DailyDialog is English-only)
        if "en" not in languages:
            return []
            
        print("Processing DailyDialog dataset...")
        
        # Process each dialogue in the training split
        for idx, dialogue in enumerate(tqdm(dataset['train'], desc="Processing dialogues")):
            utterances = dialogue['dialog']
            acts = dialogue['act']
            emotions = dialogue['emotion']
            
            # Skip if too short
            if len(utterances) < 3:
                continue
                
            # Create conversation turns
            turns = []
            transition_points = []
            transition_types = []
            topics = ["General Conversation"]  # Default topic
            
            for i, (utt, act, emotion) in enumerate(zip(utterances, acts, emotions)):
                # Detect topic transitions based on dialogue acts and emotions
                if i > 0:
                    prev_act = acts[i-1]
                    prev_emotion = emotions[i-1]
                    
                    # Topic transition heuristics:
                    # - Act sequence: inform/question after directive/commissive
                    # - Emotion changes
                    # - Presence of topic markers
                    # - Semantic shifts
                    topic_markers = self._extract_topic_markers(utt)
                    has_act_transition = (act in [1, 2] and prev_act in [3, 4])
                    has_emotion_change = (emotion != prev_emotion and emotion != 0)
                    has_semantic_shift = self._detect_semantic_shift(turns[-1].text, utt)
                    
                    if topic_markers or has_act_transition or has_emotion_change or has_semantic_shift:
                        transition_points.append(i)
                        transition_types.append("smooth" if topic_markers else "semantic")
                        
                        # Try to extract subtopic
                        subtopic = self._extract_subtopic(utt, topics[-1])
                        if subtopic:
                            topics.append(subtopic)
                
                # Create turn with metadata
                turn = ConversationTurn(
                    text=utt,
                    language="en",
                    speaker_id=str(i % 2),  # Alternate between speakers
                    topic_markers=self._extract_topic_markers(utt),
                    cultural_markers=self._extract_cultural_markers(utt, "en")
                )
                turns.append(turn)
            
            # Create conversation object
            conversation = Conversation(
                turns=turns,
                language="en",
                topics=topics,
                transition_points=transition_points,
                transition_types=transition_types,
                source="real",
                quality_score=self._compute_quality_score(turns, transition_points)
            )
            
            conversations.append(conversation)
            
            # Progress update
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} dialogues...")
        
        print(f"Processed {len(conversations)} valid conversations from DailyDialog")
        return conversations 