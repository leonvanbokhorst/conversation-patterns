import time
import math
import random
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EmotionalState:
    """Represents a complex emotional state with intensity and duration."""

    primary: Dict[str, float] = field(
        default_factory=dict
    )  # Primary emotions with intensities
    secondary: Dict[str, float] = field(
        default_factory=dict
    )  # Secondary/derived emotions
    duration: float = 0.0  # How long the emotional state persisted

    # Pre-defined emotion categories for better organization
    PRIMARY_EMOTIONS = {
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "trust",
        "anticipation",
    }

    SECONDARY_EMOTIONS = {
        "excitement",
        "nostalgia",
        "frustration",
        "anxiety",
        "contentment",
        "pride",
        "guilt",
        "shame",
        "awe",
        "gratitude",
        "envy",
        "hope",
    }

    def get_dominant_emotion(self) -> tuple[str, float]:
        """Returns the strongest emotion and its intensity."""
        all_emotions = {**self.primary, **self.secondary}
        return max(all_emotions.items(), key=lambda x: x[1], default=("neutral", 0.0))

    def emotional_energy(self) -> float:
        """Calculate overall emotional intensity."""
        return sum(self.primary.values()) * 0.7 + sum(self.secondary.values()) * 0.3


@dataclass
class MemoryNode:
    """Enhanced memory node with richer emotional and contextual features."""

    content: str
    emotional_state: EmotionalState
    timestamp: float
    embedding: np.ndarray = None  # Semantic embedding
    novelty: float = 1.0  # Initial novelty score
    tags: Set[str] = field(default_factory=set)
    connections: Dict["MemoryNode", float] = field(default_factory=dict)
    activation_level: float = 0.0
    reinforcement_count: int = 0  # Track how often this memory is recalled

    def __hash__(self):
        """Make MemoryNode hashable based on content and timestamp."""
        return hash((self.content, self.timestamp))

    def __eq__(self, other):
        """Define equality based on content and timestamp."""
        if not isinstance(other, MemoryNode):
            return False
        return self.content == other.content and self.timestamp == other.timestamp

    def decay_novelty(self, rate: float = 0.1):
        """Decay the novelty score over time."""
        self.novelty *= 1 - rate

    def reinforce(self):
        """Reinforce the memory through recall."""
        self.reinforcement_count += 1
        self.activation_level = min(1.0, self.activation_level + 0.1)


@dataclass
class NarrativeCluster:
    """Enhanced narrative cluster with dynamic coherence tracking."""

    theme: str
    memories: List[MemoryNode] = field(default_factory=list)
    emotional_signature: Dict[str, float] = field(default_factory=dict)
    coherence_score: float = 0.0
    gravity_field: float = 0.0  # Measure of cluster's "pull" on related memories

    def update_gravity(self):
        """Update the cluster's narrative gravity based on size and coherence."""
        self.gravity_field = self.coherence_score * math.log(len(self.memories) + 1)


class EnhancedMemorySystem:
    """Enhanced memory system with improved emotional and thematic processing."""

    def __init__(self):
        self.short_term_buffer: List[MemoryNode] = []
        self.long_term_storage: List[MemoryNode] = []
        self.narrative_clusters: List[NarrativeCluster] = []
        self.current_context: Dict = {}
        self.buffer_size_limit: int = 5
        self.last_consolidation_time: float = time.time()
        # Initialize semantic model
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    def process_new_experience(
        self,
        content: str,
        emotions: Dict[str, float],
        context: Dict,
        duration: float = 1.0,
    ):
        """Process a new experience with enhanced emotional differentiation."""
        # Create emotional state with primary/secondary emotion classification
        emotional_state = self._classify_emotions(emotions, duration)

        # Generate semantic embedding
        embedding = self.semantic_model.encode(content)

        # Create new memory with novelty calculation
        memory = MemoryNode(
            content=content,
            emotional_state=emotional_state,
            timestamp=time.time(),
            embedding=embedding,
            novelty=self._calculate_novelty(content, context),
        )

        # Add context-based tags
        memory.tags = self._extract_tags(content, context)

        # Add to short-term buffer with attention boost for novel/emotional content
        attention_score = memory.novelty * emotional_state.emotional_energy()
        memory.activation_level = attention_score
        self.short_term_buffer.append(memory)

        # Immediately consolidate to long-term storage for demonstration
        self.long_term_storage.append(memory)

        # Trigger resonance with attention boost
        self._activate_resonance(memory, attention_boost=attention_score)

        # Check for consolidation of other memories
        if len(self.short_term_buffer) >= self.buffer_size_limit:
            self._consolidate_memories()

    def _classify_emotions(
        self, emotions: Dict[str, float], duration: float
    ) -> EmotionalState:
        """Classify and organize emotions into primary and secondary categories."""
        emotional_state = EmotionalState(duration=duration)

        for emotion, intensity in emotions.items():
            if emotion in EmotionalState.PRIMARY_EMOTIONS:
                emotional_state.primary[emotion] = intensity
            elif emotion in EmotionalState.SECONDARY_EMOTIONS:
                emotional_state.secondary[emotion] = intensity
            else:
                # Attempt to map unknown emotion to primary/secondary
                mapped_emotion = self._map_emotion(emotion)
                if mapped_emotion in EmotionalState.PRIMARY_EMOTIONS:
                    emotional_state.primary[mapped_emotion] = intensity
                else:
                    emotional_state.secondary[mapped_emotion] = intensity

        return emotional_state

    def _calculate_novelty(self, content: str, context: Dict) -> float:
        """Calculate novelty score based on semantic and contextual uniqueness."""
        novelty_score = 1.0

        # Generate embedding for new content
        new_embedding = self.semantic_model.encode(content)

        # Calculate semantic similarities with existing memories
        if self.long_term_storage:
            similarities = cosine_similarity(
                new_embedding.reshape(1, -1),
                np.vstack([m.embedding for m in self.long_term_storage]),
            )[0]
            max_similarity = float(max(similarities))
            novelty_score *= 1 - max_similarity

        # Factor in context uniqueness
        context_tags = set(context.get("tags", []))
        context_novelty = 1.0
        if context_tags:
            for memory in self.long_term_storage:
                overlap = len(context_tags & memory.tags) / len(
                    context_tags | memory.tags
                )
                context_novelty = min(context_novelty, 1 - overlap)

        return (novelty_score + context_novelty) / 2

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity based on word overlap."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        return overlap / total if total > 0 else 0.0

    def _map_emotion(self, emotion: str) -> str:
        """Map unknown emotions to known categories based on simple word similarity."""
        # This could be enhanced with a more sophisticated emotion mapping system
        emotion_lower = emotion.lower()

        # Simple mapping rules
        joy_related = {"happy", "excited", "pleased", "delighted"}
        sadness_related = {"unhappy", "depressed", "down", "blue"}
        anger_related = {"mad", "furious", "irritated", "annoyed"}

        if emotion_lower in joy_related:
            return "joy"
        elif emotion_lower in sadness_related:
            return "sadness"
        elif emotion_lower in anger_related:
            return "anger"

        return emotion  # Keep as is if no mapping found

    def _consolidate_memories(self):
        """Enhanced memory consolidation with emotional reinforcement and memory merging."""
        consolidation_threshold = 0.2  # Lower threshold for demonstration
        merge_threshold = 0.8  # Threshold for memory merging

        # First pass: Check for memories to merge
        merged_memories = set()
        i = 0
        while i < len(self.long_term_storage):
            memory1 = self.long_term_storage[i]
            if memory1 in merged_memories:
                i += 1
                continue

            j = i + 1
            while j < len(self.long_term_storage):
                memory2 = self.long_term_storage[j]
                if memory2 in merged_memories:
                    j += 1
                    continue

                # Calculate overall similarity
                resonance = self._calculate_resonance(memory1, memory2)

                if resonance > merge_threshold:
                    # Create concise merged content
                    content_parts = set()
                    for content in [memory1.content, memory2.content]:
                        # Split compound content if it exists
                        parts = [p.strip() for p in content.split(";")]
                        content_parts.update(parts)

                    # Sort parts for consistent ordering
                    merged_content = "; ".join(sorted(content_parts))

                    # Combine emotional states
                    merged_emotions = {}
                    for emotion, intensity in memory1.emotional_state.primary.items():
                        merged_emotions[emotion] = intensity
                    for emotion, intensity in memory2.emotional_state.primary.items():
                        if emotion in merged_emotions:
                            merged_emotions[emotion] = max(
                                merged_emotions[emotion], intensity
                            )
                        else:
                            merged_emotions[emotion] = intensity

                    # Create merged memory
                    merged_memory = MemoryNode(
                        content=merged_content,
                        emotional_state=self._classify_emotions(
                            merged_emotions,
                            max(
                                memory1.emotional_state.duration,
                                memory2.emotional_state.duration,
                            ),
                        ),
                        timestamp=max(memory1.timestamp, memory2.timestamp),
                        embedding=self.semantic_model.encode(merged_content),
                        novelty=min(memory1.novelty, memory2.novelty),
                        tags=memory1.tags | memory2.tags,
                        activation_level=max(
                            memory1.activation_level, memory2.activation_level
                        ),
                        reinforcement_count=memory1.reinforcement_count
                        + memory2.reinforcement_count,
                    )

                    # Transfer connections safely
                    connections_to_transfer = []

                    # Collect connections from memory1
                    for connected_memory, strength in list(memory1.connections.items()):
                        if (
                            connected_memory not in merged_memories
                            and connected_memory != memory2
                        ):
                            connections_to_transfer.append((connected_memory, strength))

                    # Collect connections from memory2
                    for connected_memory, strength in list(memory2.connections.items()):
                        if (
                            connected_memory not in merged_memories
                            and connected_memory != memory1
                        ):
                            # Check if we already have this connection
                            existing_connection = next(
                                (
                                    c
                                    for c, s in connections_to_transfer
                                    if c == connected_memory
                                ),
                                None,
                            )
                            if existing_connection:
                                # Update existing connection with max strength
                                idx = next(
                                    i
                                    for i, (c, s) in enumerate(connections_to_transfer)
                                    if c == connected_memory
                                )
                                connections_to_transfer[idx] = (
                                    connected_memory,
                                    max(connections_to_transfer[idx][1], strength),
                                )
                            else:
                                connections_to_transfer.append(
                                    (connected_memory, strength)
                                )

                    # Apply collected connections
                    for connected_memory, strength in connections_to_transfer:
                        merged_memory.connections[connected_memory] = strength
                        connected_memory.connections[merged_memory] = strength

                    # Mark memories for removal
                    merged_memories.add(memory1)
                    merged_memories.add(memory2)

                    # Add merged memory to storage
                    self.long_term_storage.append(merged_memory)

                    print(f"Merged memories: {memory1.content} + {memory2.content}")
                j += 1
            i += 1

        # Remove merged memories safely
        self.long_term_storage = [
            m for m in self.long_term_storage if m not in merged_memories
        ]

        # Second pass: Normal consolidation for remaining memories
        for memory in list(self.short_term_buffer):  # Create a copy for safe iteration
            if memory in merged_memories:
                continue

            # Emotional intensity influences consolidation chance
            emotional_intensity = memory.emotional_state.emotional_energy()
            consolidation_chance = 0.9 + (
                0.1 * emotional_intensity
            )  # Higher base chance

            if (
                random.random() < consolidation_chance
                or memory.activation_level > consolidation_threshold
            ):
                if memory not in self.long_term_storage:  # Avoid duplicates
                    self.long_term_storage.append(memory)
                self._integrate_with_narratives(memory)

        self.short_term_buffer.clear()

        # Update narrative clusters after consolidation
        self._update_narrative_clusters()

    def _integrate_with_narratives(self, memory: MemoryNode):
        """Integrate a memory into narrative clusters with enhanced thematic coherence."""
        best_cluster: Optional[NarrativeCluster] = None
        best_fit_score = 0.0

        # Find best fitting cluster
        for cluster in self.narrative_clusters:
            fit_score = self._calculate_cluster_fit(memory, cluster)
            if fit_score > best_fit_score:
                best_fit_score = fit_score
                best_cluster = cluster

        # Create new cluster if no good fit found (lowered threshold)
        if best_fit_score < 0.2:  # Lower threshold for cluster creation
            theme = self._extract_semantic_theme([memory])
            new_cluster = NarrativeCluster(theme=theme)
            new_cluster.memories.append(memory)
            self.narrative_clusters.append(new_cluster)
            new_cluster.update_gravity()
        else:
            best_cluster.memories.append(memory)
            # Update cluster theme based on all memories
            best_cluster.theme = self._extract_semantic_theme(best_cluster.memories)
            best_cluster.update_gravity()
            self._update_cluster_coherence(best_cluster)

    def _calculate_cluster_fit(
        self, memory: MemoryNode, cluster: NarrativeCluster
    ) -> float:
        """Calculate how well a memory fits into a cluster using semantic and emotional factors."""
        if not cluster.memories:
            return 0.0

        # Calculate semantic similarity with cluster memories
        cluster_embeddings = np.vstack([m.embedding for m in cluster.memories])
        semantic_similarities = cosine_similarity(
            memory.embedding.reshape(1, -1), cluster_embeddings
        )[0]
        semantic_fit = float(np.mean(semantic_similarities))

        # Calculate emotional congruence
        emotional_congruence = self._calculate_emotional_congruence(
            memory.emotional_state, cluster
        )

        # Calculate thematic similarity (tags)
        theme_similarity = len(
            memory.tags & set().union(*(m.tags for m in cluster.memories))
        ) / len(memory.tags | set().union(*(m.tags for m in cluster.memories)))

        # Weight the factors
        return (
            0.4 * semantic_fit  # Semantic similarity
            + 0.3 * emotional_congruence  # Emotional congruence
            + 0.3 * theme_similarity  # Thematic similarity
        )

    def _extract_semantic_theme(self, memories: List[MemoryNode]) -> str:
        """Extract a theme from memories using semantic analysis and emotional context."""
        if not memories:
            return "Miscellaneous"

        # Get the centroid embedding
        embeddings = np.vstack([m.embedding for m in memories])
        centroid = np.mean(embeddings, axis=0)

        # Find the memory closest to the centroid
        distances = cosine_similarity(centroid.reshape(1, -1), embeddings)[0]
        central_memory = memories[int(np.argmax(distances))]

        # Get emotional context
        emotional_intensities = []
        for memory in memories:
            emotional_intensities.extend(
                list(memory.emotional_state.primary.values())
                + list(memory.emotional_state.secondary.values())
            )
        if emotional_intensities:
            avg_emotional_intensity = sum(emotional_intensities) / len(
                emotional_intensities
            )
        else:
            avg_emotional_intensity = 0.0

        # Extract key terms from central memory
        key_terms = set()
        for memory in memories:
            key_terms.update(
                word.capitalize()
                for word in memory.content.lower().split()
                if len(word) > 3
                and word not in {"with", "that", "this", "from", "were", "what"}
            )

        # Get the most common tags
        all_tags = []
        for memory in memories:
            all_tags.extend(
                tag
                for tag in memory.tags
                if not any(tag.lower() in term.lower() for term in key_terms)
            )

        if all_tags:
            from collections import Counter

            most_common_tag = Counter(all_tags).most_common(1)[0][0].capitalize()
            key_terms.add(most_common_tag)

        # Get dominant emotion
        all_emotions = {}
        for memory in memories:
            for emotion, intensity in memory.emotional_state.primary.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + intensity
            for emotion, intensity in memory.emotional_state.secondary.items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + intensity

        if all_emotions:
            dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
            # Only include emotion if it's significant
            if all_emotions[dominant_emotion] / len(memories) > 0.5:
                return f"{' & '.join(sorted(key_terms)[:2])} ({dominant_emotion})"

        return f"{' & '.join(sorted(key_terms)[:2])}"

    def _calculate_emotional_congruence(
        self, emotional_state: EmotionalState, cluster: NarrativeCluster
    ) -> float:
        """Calculate emotional similarity between a memory and a cluster."""
        if not cluster.emotional_signature:
            return 1.0

        total_diff = 0.0
        count = 0

        # Compare primary emotions
        for emotion in EmotionalState.PRIMARY_EMOTIONS:
            if (
                emotion in cluster.emotional_signature
                or emotion in emotional_state.primary
            ):
                cluster_val = cluster.emotional_signature.get(emotion, 0.0)
                memory_val = emotional_state.primary.get(emotion, 0.0)
                total_diff += abs(cluster_val - memory_val)
                count += 1

        # Compare secondary emotions
        for emotion in EmotionalState.SECONDARY_EMOTIONS:
            if (
                emotion in cluster.emotional_signature
                or emotion in emotional_state.secondary
            ):
                cluster_val = cluster.emotional_signature.get(emotion, 0.0)
                memory_val = emotional_state.secondary.get(emotion, 0.0)
                total_diff += abs(cluster_val - memory_val)
                count += 1

        return 1.0 if count == 0 else 1.0 - (total_diff / count)

    def _update_cluster_coherence(self, cluster: NarrativeCluster):
        """Update cluster coherence based on memory relationships and emotional consistency."""
        if len(cluster.memories) < 2:
            cluster.coherence_score = 1.0
            return

        # Calculate average connection strength
        connection_scores = []
        for i, mem1 in enumerate(cluster.memories):
            connection_scores.extend(
                mem1.connections[mem2]
                for mem2 in cluster.memories[i + 1 :]
                if mem2 in mem1.connections
            )
        # Calculate emotional consistency
        emotional_variance = self._calculate_emotional_variance(cluster)

        # Combine metrics
        connection_strength = (
            sum(connection_scores) / len(connection_scores)
            if connection_scores
            else 0.0
        )
        cluster.coherence_score = 0.7 * connection_strength + 0.3 * (
            1 - emotional_variance
        )

        # Update cluster's emotional signature
        self._update_cluster_emotional_signature(cluster)

    def _calculate_emotional_variance(self, cluster: NarrativeCluster) -> float:
        """Calculate the emotional variance within a cluster."""
        if len(cluster.memories) < 2:
            return 0.0

        all_emotions = set()
        for memory in cluster.memories:
            all_emotions.update(memory.emotional_state.primary.keys())
            all_emotions.update(memory.emotional_state.secondary.keys())

        total_variance = 0.0
        for emotion in all_emotions:
            values = []
            for memory in cluster.memories:
                val = memory.emotional_state.primary.get(
                    emotion, 0.0
                ) + memory.emotional_state.secondary.get(emotion, 0.0)
                if val > 0:
                    values.append(val)

            if values:
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                total_variance += variance

        return total_variance / len(all_emotions) if all_emotions else 0.0

    def _update_cluster_emotional_signature(self, cluster: NarrativeCluster):
        """Update the emotional signature of a cluster based on its memories."""
        if not cluster.memories:
            return

        new_signature = {}

        # Combine all emotions from memories
        for memory in cluster.memories:
            for emotion, intensity in memory.emotional_state.primary.items():
                new_signature[emotion] = new_signature.get(emotion, 0.0) + intensity
            for emotion, intensity in memory.emotional_state.secondary.items():
                new_signature[emotion] = new_signature.get(emotion, 0.0) + intensity

        # Average the intensities
        num_memories = len(cluster.memories)
        cluster.emotional_signature = {
            emotion: value / num_memories for emotion, value in new_signature.items()
        }

    def _extract_tags(self, content: str, context: Dict) -> Set[str]:
        """Extract tags from content and context for memory organization."""
        tags = set()

        if context:
            if "tags" in context:
                tags.update(context["tags"])

            if "location" in context:
                tags.add(context["location"])

            if "activity" in context:
                tags.add(context["activity"])

            if "social" in context:
                tags.add(context["social"])

        # Extract key terms from content (simple word-based approach)
        # In a real implementation, this could use NLP for better extraction
        words = set(content.lower().split())
        important_words = {
            word
            for word in words
            if len(word) > 3  # Skip short words
            and word not in {"with", "that", "this", "from", "were", "what"}
        }  # Skip common words
        tags.update(important_words)

        return tags

    def _extract_theme_from_memory(self, memory: MemoryNode) -> str:
        """Extract a theme from a memory based on its tags and emotional content."""
        # Get the most frequent tags across connected memories
        connected_tags = {}
        for connected_memory in memory.connections.keys():
            for tag in connected_memory.tags:
                connected_tags[tag] = connected_tags.get(tag, 0) + 1

        # Add current memory's tags
        for tag in memory.tags:
            connected_tags[tag] = (
                connected_tags.get(tag, 0) + 2
            )  # Weight current memory's tags more

        # Get emotional context
        dominant_emotion, _ = memory.emotional_state.get_dominant_emotion()

        # Find most common tag
        if connected_tags:
            primary_theme = max(connected_tags.items(), key=lambda x: x[1])[0]
        else:
            # If no connected memories, use a tag from current memory
            primary_theme = next(iter(memory.tags)) if memory.tags else "Miscellaneous"

        # Combine with emotional context for richer theme
        if dominant_emotion != "neutral":
            return f"{primary_theme.capitalize()} ({dominant_emotion})"
        else:
            return primary_theme.capitalize()

    def _activate_resonance(
        self, trigger_memory: MemoryNode, attention_boost: float = 0.0
    ):
        """Activate related memories based on content and emotional similarity."""
        activation_scores = {}

        # Calculate resonance with all long-term memories
        for memory in self.long_term_storage:
            score = self._calculate_resonance(trigger_memory, memory)

            # Apply attention boost for novel or emotionally significant memories
            score += attention_boost * memory.novelty

            if score > 0.3:  # Activation threshold
                memory.activation_level = score
                activation_scores[memory] = score

                # Create or strengthen connections
                if memory not in trigger_memory.connections:
                    trigger_memory.connections[memory] = score
                    memory.connections[trigger_memory] = score
                else:
                    # Strengthen existing connection
                    current_strength = trigger_memory.connections[memory]
                    new_strength = min(1.0, current_strength + score * 0.2)
                    trigger_memory.connections[memory] = new_strength
                    memory.connections[trigger_memory] = new_strength

        # Update narrative clusters
        self._update_narrative_clusters()

        # Decay novelty for activated memories
        for memory in activation_scores:
            memory.decay_novelty()

    def _update_narrative_clusters(self):
        """Update narrative clusters based on current memory activations."""
        # Skip if no memories
        if not self.long_term_storage:
            return

        # Get currently activated memories
        activated_memories = [
            m for m in self.long_term_storage if m.activation_level > 0.3
        ]

        # Update existing clusters
        for cluster in self.narrative_clusters:
            # Check if any activated memories belong to this cluster
            cluster_activated = any(m in cluster.memories for m in activated_memories)
            if cluster_activated:
                self._update_cluster_coherence(cluster)
                cluster.update_gravity()

        if unclustered_memories := [
            m
            for m in activated_memories
            if all(m not in c.memories for c in self.narrative_clusters)
        ]:
            # Try to form new clusters from unclustered memories
            while unclustered_memories:
                seed_memory = unclustered_memories[0]
                if related_memories := [
                    m
                    for m in unclustered_memories[1:]
                    if seed_memory in m.connections or m in seed_memory.connections
                ]:
                    # Create new cluster
                    new_cluster = NarrativeCluster(
                        theme=self._extract_theme_from_memory(seed_memory)
                    )
                    new_cluster.memories.extend([seed_memory] + related_memories)
                    self._update_cluster_coherence(new_cluster)
                    new_cluster.update_gravity()
                    self.narrative_clusters.append(new_cluster)

                    # Remove clustered memories from unclustered list
                    for memory in related_memories + [seed_memory]:
                        if memory in unclustered_memories:
                            unclustered_memories.remove(memory)
                else:
                    # No related memories found, remove seed memory
                    unclustered_memories.remove(seed_memory)

    def _calculate_resonance(self, memory1: MemoryNode, memory2: MemoryNode) -> float:
        """Calculate resonance between two memories based on multiple factors."""
        if memory1 == memory2:
            return 0.0

        # Calculate semantic similarity with more weight
        semantic_similarity = float(
            cosine_similarity(
                memory1.embedding.reshape(1, -1), memory2.embedding.reshape(1, -1)
            )[0, 0]
        )

        # Calculate emotional resonance
        emotional_resonance = self._calculate_emotional_congruence(
            memory1.emotional_state,
            NarrativeCluster(
                theme="temp", emotional_signature=memory2.emotional_state.primary
            ),
        )

        # Calculate contextual overlap (tags) with more emphasis on important tags
        important_tags = {"work", "project", "meeting", "achievement", "milestone"}
        tag_overlap_base = len(memory1.tags & memory2.tags)
        important_overlap = len((memory1.tags & memory2.tags) & important_tags)
        tag_overlap = (tag_overlap_base + 2 * important_overlap) / max(
            len(memory1.tags | memory2.tags), 1
        )

        # Calculate temporal proximity (decay over time) - more gradual decay
        time_diff = abs(memory1.timestamp - memory2.timestamp)
        temporal_factor = math.exp(-time_diff / (7 * 24 * 3600))  # 7-day decay

        # Weight the factors with emphasis on semantic and emotional connections
        resonance = (
            0.4 * semantic_similarity  # Increased weight for semantic similarity
            + 0.3 * emotional_resonance  # Emotional resonance
            + 0.2 * tag_overlap  # Contextual overlap
            + 0.1 * temporal_factor  # Temporal proximity
        )

        # Boost connections for highly similar or emotionally resonant memories
        boost = 1.0
        if semantic_similarity > 0.8 or emotional_resonance > 0.8:
            boost = 1.3

        # Ensure minimum connection strength for visualization
        base_connection = 0.2
        return max(base_connection, min(1.0, resonance * boost))
