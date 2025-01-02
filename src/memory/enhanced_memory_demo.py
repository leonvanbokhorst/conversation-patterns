from enhanced_memory_system import EnhancedMemorySystem
import time


def demonstrate_emotional_differentiation():
    """Demonstrate the enhanced emotional differentiation capabilities."""
    print("\n=== Demonstrating Enhanced Emotional Processing ===")

    memory_system = EnhancedMemorySystem()

    # Complex emotional experiences
    experiences = [
        {
            "content": "Gave a presentation at work that received mixed feedback",
            "emotions": {"anxiety": 0.7, "hope": 0.6, "pride": 0.5, "frustration": 0.3},
            "context": {
                "location": "Office",
                "activity": "presentation",
                "social": "group",
                "tags": ["work", "presentation", "feedback"],
            },
        },
        {
            "content": "Had a deep conversation with an old friend about life changes",
            "emotions": {
                "joy": 0.8,
                "nostalgia": 0.7,
                "trust": 0.9,
                "anticipation": 0.4,
            },
            "context": {
                "location": "Cafe",
                "activity": "conversation",
                "social": "friend",
                "tags": ["friendship", "life_changes", "conversation"],
            },
        },
        {
            "content": "Watched a powerful documentary about climate change",
            "emotions": {
                "sadness": 0.6,
                "anger": 0.5,
                "fear": 0.4,
                "hope": 0.3,
                "awe": 0.8,
            },
            "context": {
                "location": "Home",
                "activity": "watching",
                "social": "solo",
                "tags": ["documentary", "climate", "learning"],
            },
        },
    ]

    # Process experiences
    for exp in experiences:
        print(f"\nProcessing experience: {exp['content']}")
        memory_system.process_new_experience(**exp)
        time.sleep(0.1)  # Simulate time passing

        # Show emotional classification
        latest_memory = memory_system.short_term_buffer[-1]
        print("\nEmotional Classification:")
        print("Primary Emotions:", latest_memory.emotional_state.primary)
        print("Secondary Emotions:", latest_memory.emotional_state.secondary)
        print(
            f"Emotional Energy: {latest_memory.emotional_state.emotional_energy():.2f}"
        )
        dominant_emotion, intensity = (
            latest_memory.emotional_state.get_dominant_emotion()
        )
        print(f"Dominant Emotion: {dominant_emotion} ({intensity:.2f})")


def demonstrate_thematic_coherence():
    """Demonstrate the enhanced thematic coherence and narrative clustering."""
    print("\n=== Demonstrating Thematic Coherence ===")

    memory_system = EnhancedMemorySystem()

    # Series of related experiences
    experience_series = [
        # Learning Series
        {
            "content": "Started learning Python programming",
            "emotions": {"excitement": 0.8, "anticipation": 0.7, "joy": 0.6},
            "context": {
                "location": "Home",
                "activity": "learning",
                "tags": ["programming", "learning", "technology"],
            },
        },
        {
            "content": "Completed first coding project",
            "emotions": {"pride": 0.9, "joy": 0.8, "satisfaction": 0.7},
            "context": {
                "location": "Home",
                "activity": "coding",
                "tags": ["programming", "achievement", "technology"],
            },
        },
        # Nature Series
        {
            "content": "Went hiking in the mountains",
            "emotions": {"awe": 0.9, "joy": 0.7, "peace": 0.8},
            "context": {
                "location": "Mountains",
                "activity": "hiking",
                "tags": ["nature", "exercise", "outdoors"],
            },
        },
        {
            "content": "Watched sunset at the beach",
            "emotions": {"peace": 0.9, "awe": 0.8, "contentment": 0.7},
            "context": {
                "location": "Beach",
                "activity": "relaxation",
                "tags": ["nature", "sunset", "outdoors"],
            },
        },
    ]

    # Process experiences
    for exp in experience_series:
        print(f"\nAdding experience: {exp['content']}")
        memory_system.process_new_experience(**exp)
        time.sleep(0.1)

    # Examine narrative clusters
    print("\nNarrative Clusters Formed:")
    for i, cluster in enumerate(memory_system.narrative_clusters, 1):
        print(f"\nCluster {i}: {cluster.theme}")
        print("Memories:")
        for memory in cluster.memories:
            print(f"- {memory.content}")
        print(f"Coherence Score: {cluster.coherence_score:.2f}")
        print(f"Gravity Field: {cluster.gravity_field:.2f}")
        print(
            "Emotional Signature:",
            {k: f"{v:.2f}" for k, v in cluster.emotional_signature.items()},
        )


def demonstrate_memory_resonance():
    """Demonstrate how memories resonate and connect based on emotions and themes."""
    print("\n=== Demonstrating Memory Resonance ===")

    memory_system = EnhancedMemorySystem()

    # Add some initial memories
    base_experiences = [
        {
            "content": "First day at new job",
            "emotions": {"anxiety": 0.7, "excitement": 0.8, "anticipation": 0.6},
            "context": {
                "location": "Office",
                "activity": "work",
                "tags": ["work", "new_beginning", "career"],
            },
        },
        {
            "content": "Team celebration after project success",
            "emotions": {"joy": 0.9, "pride": 0.8, "gratitude": 0.7},
            "context": {
                "location": "Office",
                "activity": "celebration",
                "tags": ["work", "achievement", "team"],
            },
        },
    ]

    # Add base experiences
    for exp in base_experiences:
        memory_system.process_new_experience(**exp)
        time.sleep(0.1)

    # Now add a resonant experience
    resonant_experience = {
        "content": "Leading first team meeting in new role",
        "emotions": {"anxiety": 0.6, "excitement": 0.7, "pride": 0.5},
        "context": {
            "location": "Office",
            "activity": "meeting",
            "tags": ["work", "leadership", "team"],
        },
    }

    print("\nAdding resonant experience:", resonant_experience["content"])
    memory_system.process_new_experience(**resonant_experience)

    # Show activation levels and connections
    print("\nMemory Activation Levels:")
    for memory in memory_system.long_term_storage:
        print(f"\nMemory: {memory.content}")
        print(f"Activation Level: {memory.activation_level:.2f}")
        print("Connected Memories:")
        for connected_memory, strength in memory.connections.items():
            print(f"- {connected_memory.content} (strength: {strength:.2f})")


def demonstrate_visualization():
    """Demonstrate the visualization capabilities of the memory system."""
    print("\n=== Demonstrating Memory Visualization ===")

    from memory_visualizer import MemoryVisualizer

    memory_system = EnhancedMemorySystem()
    visualizer = MemoryVisualizer()

    # Create a rich set of interconnected memories
    experiences = [
        {
            "content": "Started a new project at work",
            "emotions": {"excitement": 0.8, "anticipation": 0.7, "anxiety": 0.4},
            "context": {
                "location": "Office",
                "activity": "work",
                "tags": ["work", "project", "beginning"],
            },
        },
        {
            "content": "Team brainstorming session",
            "emotions": {"joy": 0.7, "trust": 0.8, "anticipation": 0.6},
            "context": {
                "location": "Conference Room",
                "activity": "meeting",
                "tags": ["work", "collaboration", "creativity"],
            },
        },
        {
            "content": "Solved a complex technical problem",
            "emotions": {"pride": 0.9, "joy": 0.8, "satisfaction": 0.7},
            "context": {
                "location": "Office",
                "activity": "problem_solving",
                "tags": ["work", "achievement", "technical"],
            },
        },
        {
            "content": "Project presentation to stakeholders",
            "emotions": {"anxiety": 0.6, "excitement": 0.7, "anticipation": 0.8},
            "context": {
                "location": "Board Room",
                "activity": "presentation",
                "tags": ["work", "communication", "milestone"],
            },
        },
    ]

    # Process experiences
    print("\nProcessing experiences for visualization...")
    for exp in experiences:
        memory_system.process_new_experience(**exp)
        time.sleep(0.1)

    # Debug information
    print(f"\nLong-term storage size: {len(memory_system.long_term_storage)}")
    print(f"Number of narrative clusters: {len(memory_system.narrative_clusters)}")

    if memory_system.long_term_storage:
        print("\nMemory connections:")
        for memory in memory_system.long_term_storage:
            print(f"\nMemory: {memory.content}")
            print(f"Connections: {len(memory.connections)}")
            print(f"Tags: {memory.tags}")
            print(f"Activation level: {memory.activation_level:.2f}")

    # Create visualizations
    print("\nGenerating memory network visualization...")
    visualizer.visualize_memory_network(
        memory_system.long_term_storage, save_path="memory_network.png"
    )

    print("\nGenerating narrative clusters visualization...")
    if memory_system.narrative_clusters:
        visualizer.visualize_narrative_clusters(
            memory_system.narrative_clusters, save_path="narrative_clusters.png"
        )
    else:
        print("No narrative clusters to visualize.")

    print("\nGenerating emotion legend...")
    visualizer.create_emotion_legend(save_path="emotion_legend.png")

    print("\nVisualizations have been saved to:")
    print("- memory_network.png")
    if memory_system.narrative_clusters:
        print("- narrative_clusters.png")
    print("- emotion_legend.png")


def demonstrate_memory_consolidation():
    """Demonstrate adaptive memory consolidation with merging of similar experiences."""
    print("\n=== Demonstrating Memory Consolidation ===")

    memory_system = EnhancedMemorySystem()

    # Add similar experiences about learning and practice
    experiences = [
        {
            "content": "Started learning Python basics",
            "emotions": {"excitement": 0.7, "anticipation": 0.6},
            "context": {
                "location": "Home",
                "activity": "learning",
                "tags": ["programming", "learning", "python"],
            },
        },
        {
            "content": "Practiced Python fundamentals",
            "emotions": {"excitement": 0.6, "satisfaction": 0.5},
            "context": {
                "location": "Home",
                "activity": "practice",
                "tags": ["programming", "practice", "python"],
            },
        },
        {
            "content": "Completed first coding exercise",
            "emotions": {"pride": 0.8, "joy": 0.7},
            "context": {
                "location": "Home",
                "activity": "coding",
                "tags": ["programming", "achievement", "python"],
            },
        },
    ]

    # Process experiences with time gaps
    for exp in experiences:
        print(f"\nProcessing experience: {exp['content']}")
        memory_system.process_new_experience(**exp)
        time.sleep(0.1)  # Simulate time passing

    print("\nMemory consolidation status:")
    print(f"Long-term storage size: {len(memory_system.long_term_storage)}")
    print(f"Number of narrative clusters: {len(memory_system.narrative_clusters)}")

    # Add more experiences in a different context
    work_experiences = [
        {
            "content": "Morning team standup meeting",
            "emotions": {"neutral": 0.5, "anticipation": 0.3},
            "context": {
                "location": "Office",
                "activity": "meeting",
                "tags": ["work", "team", "communication"],
            },
        },
        {
            "content": "Daily team sync discussion",
            "emotions": {"neutral": 0.4, "trust": 0.4},
            "context": {
                "location": "Office",
                "activity": "meeting",
                "tags": ["work", "team", "communication"],
            },
        },
    ]

    print("\nAdding work experiences...")
    for exp in work_experiences:
        print(f"\nProcessing experience: {exp['content']}")
        memory_system.process_new_experience(**exp)
        time.sleep(0.1)

    print("\nFinal memory state:")
    print(f"Long-term storage size: {len(memory_system.long_term_storage)}")
    print(f"Number of narrative clusters: {len(memory_system.narrative_clusters)}")

    # Show memory contents
    print("\nMemory contents:")
    for memory in memory_system.long_term_storage:
        print(f"\nContent: {memory.content}")
        print(f"Tags: {memory.tags}")
        print(f"Activation level: {memory.activation_level:.2f}")
        if memory.connections:
            print("Connected to:")
            for connected_memory, strength in memory.connections.items():
                print(f"- {connected_memory.content} (strength: {strength:.2f})")


if __name__ == "__main__":
    print("=== Enhanced Memory System Demonstration ===")

    # Run demonstrations
    demonstrate_emotional_differentiation()
    demonstrate_thematic_coherence()
    demonstrate_memory_resonance()
    demonstrate_memory_consolidation()
    demonstrate_visualization()

    print("\n=== Demonstration Complete ===")
