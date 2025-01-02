import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
import math
import numpy as np
from enhanced_memory_system import MemoryNode, NarrativeCluster
from sklearn.metrics.pairwise import cosine_similarity


class MemoryVisualizer:
    """Visualizes memory networks and clusters using NetworkX."""

    def __init__(self):
        self.node_colors = {
            "joy": "#FFD700",  # Gold
            "sadness": "#4169E1",  # Royal Blue
            "anger": "#FF4500",  # Red-Orange
            "fear": "#800080",  # Purple
            "surprise": "#98FB98",  # Pale Green
            "trust": "#40E0D0",  # Turquoise
            "anticipation": "#FFA07A",  # Light Salmon
            "neutral": "#A9A9A9",  # Dark Gray
        }

        # Color gradients for semantic similarity
        self.semantic_colors = (
            plt.cm.viridis
        )  # Use viridis colormap for semantic relationships

    def visualize_memory_network(
        self, memories: List[MemoryNode], save_path: str = None
    ):
        """Create a network visualization of memories and their connections."""
        if not memories:
            print("No memories to visualize.")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(20, 15))

        G = nx.Graph()

        # Calculate semantic similarities between all memories
        embeddings = np.vstack([m.embedding for m in memories])
        similarity_matrix = cosine_similarity(embeddings)

        # Add nodes with larger base size
        for i, memory in enumerate(memories):
            dominant_emotion, _ = memory.emotional_state.get_dominant_emotion()

            # Calculate average semantic similarity for this memory
            avg_similarity = np.mean(similarity_matrix[i])

            G.add_node(
                memory.content,
                color=self.node_colors.get(dominant_emotion, "#A9A9A9"),
                size=4000 * (memory.activation_level + avg_similarity) + 2000,
            )

        # Add edges with semantic similarity information
        edge_count = 0
        for i, memory1 in enumerate(memories):
            for j, memory2 in enumerate(memories[i + 1 :], i + 1):
                similarity = similarity_matrix[i, j]
                if similarity > 0.3:  # Threshold for semantic similarity
                    # Blend connection strength with semantic similarity
                    connection_strength = memory1.connections.get(memory2, 0.0)
                    blended_strength = 0.7 * similarity + 0.3 * connection_strength

                    if blended_strength > 0.2:  # Lower threshold for visibility
                        G.add_edge(
                            memory1.content,
                            memory2.content,
                            weight=blended_strength,
                            color=self.semantic_colors(blended_strength),
                            alpha=min(1.0, blended_strength + 0.3),
                        )
                        edge_count += 1

        print(f"Created graph with {len(G.nodes())} nodes and {edge_count} edges")

        pos = nx.spring_layout(G, k=3, iterations=100)

        # Draw nodes
        node_colors = [G.nodes[node]["color"] for node in G.nodes()]
        node_sizes = [G.nodes[node]["size"] for node in G.nodes()]
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax,
        )

        # Draw edges with semantic colors
        if G.edges():
            edges = G.edges()
            edge_colors = [G[u][v]["color"] for u, v in edges]
            edge_alphas = [G[u][v]["alpha"] for u, v in edges]
            edge_weights = [G[u][v]["weight"] * 3 for u, v in edges]

            nx.draw_networkx_edges(
                G,
                pos,
                edge_color=edge_colors,
                width=edge_weights,
                alpha=edge_alphas,
                ax=ax,
            )

        # Add labels with better visibility
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

        # Add a colorbar to show semantic similarity scale
        sm = plt.cm.ScalarMappable(cmap=self.semantic_colors, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(
            sm, ax=ax, label="Semantic Similarity", orientation="horizontal", pad=0.1
        )

        ax.set_title(
            "Memory Network Visualization\nNode size: Activation + Semantic Centrality\nEdge color: Semantic Similarity",
            pad=20,
            fontsize=16,
        )
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        else:
            plt.show()

        plt.close()

    def visualize_narrative_clusters(
        self, clusters: List[NarrativeCluster], save_path: str = None
    ):
        """Create a visualization of narrative clusters and their relationships."""
        if not clusters:
            print("No clusters to visualize.")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(24, 16))

        G = nx.Graph()
        print(f"Visualizing {len(clusters)} clusters")

        # Calculate semantic similarities between clusters
        cluster_embeddings = []
        for cluster in clusters:
            cluster_embedding = np.mean([m.embedding for m in cluster.memories], axis=0)
            cluster_embeddings.append(cluster_embedding)

        cluster_similarities = cosine_similarity(cluster_embeddings)

        # Create a larger node for each cluster
        for i, cluster in enumerate(clusters):
            print(f"\nCluster {i+1}:")
            print(f"Theme: {cluster.theme}")
            print(f"Number of memories: {len(cluster.memories)}")
            print(f"Coherence score: {cluster.coherence_score:.2f}")

            # Calculate cluster node size
            avg_semantic_centrality = np.mean(cluster_similarities[i])
            size = (
                6000
                * math.log(len(cluster.memories) + 1)
                * (cluster.coherence_score + avg_semantic_centrality)
                + 3000
            )

            # Get dominant emotion
            cluster_emotions = cluster.emotional_signature
            dominant_emotion = (
                max(cluster_emotions.items(), key=lambda x: x[1])[0]
                if cluster_emotions
                else "neutral"
            )

            G.add_node(
                f"Cluster: {cluster.theme}",
                size=size,
                color=self.node_colors.get(dominant_emotion, "#A9A9A9"),
            )

            # Add memory nodes
            cluster_memories = []
            for memory in cluster.memories:
                dominant_emotion, _ = memory.emotional_state.get_dominant_emotion()
                memory_name = memory.content
                G.add_node(
                    memory_name,
                    size=3000 * memory.activation_level + 1500,
                    color=self.node_colors.get(dominant_emotion, "#A9A9A9"),
                )
                G.add_edge(
                    f"Cluster: {cluster.theme}",
                    memory_name,
                    weight=memory.activation_level,
                    color=self.semantic_colors(0.7),
                )
                cluster_memories.append(memory_name)

        # Add edges between similar clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                similarity = cluster_similarities[i, j]
                if similarity > 0.4:
                    G.add_edge(
                        f"Cluster: {clusters[i].theme}",
                        f"Cluster: {clusters[j].theme}",
                        weight=similarity,
                        color=self.semantic_colors(similarity),
                        alpha=similarity,
                    )

        print(f"\nCreated graph with {len(G.nodes())} nodes")

        pos = nx.spring_layout(G, k=3.5, iterations=100)

        # Draw cluster nodes
        cluster_nodes = [node for node in G.nodes() if node.startswith("Cluster:")]
        cluster_colors = [G.nodes[node]["color"] for node in cluster_nodes]
        cluster_sizes = [G.nodes[node]["size"] for node in cluster_nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=cluster_nodes,
            node_color=cluster_colors,
            node_size=cluster_sizes,
            alpha=0.8,
            ax=ax,
        )

        # Draw memory nodes
        memory_nodes = [node for node in G.nodes() if not node.startswith("Cluster:")]
        memory_colors = [G.nodes[node]["color"] for node in memory_nodes]
        memory_sizes = [G.nodes[node]["size"] for node in memory_nodes]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=memory_nodes,
            node_color=memory_colors,
            node_size=memory_sizes,
            alpha=0.9,
            ax=ax,
        )

        # Draw edges with semantic colors
        edges = G.edges()
        edge_colors = [G[u][v].get("color", "gray") for u, v in edges]
        edge_weights = [G[u][v]["weight"] * 3 for u, v in edges]
        edge_alphas = [G[u][v].get("alpha", 0.5) for u, v in edges]

        nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, width=edge_weights, alpha=edge_alphas, ax=ax
        )

        # Add labels
        cluster_labels = {node: node for node in cluster_nodes}
        memory_labels = {node: node for node in memory_nodes}

        nx.draw_networkx_labels(
            G, pos, labels=cluster_labels, font_size=12, font_weight="bold", ax=ax
        )
        nx.draw_networkx_labels(G, pos, labels=memory_labels, font_size=10, ax=ax)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.semantic_colors, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(
            sm, ax=ax, label="Semantic Similarity", orientation="horizontal", pad=0.1
        )

        ax.set_title(
            "Narrative Clusters Visualization\nNode size: Cluster size + Coherence + Semantic Centrality\nEdge color: Semantic Similarity",
            pad=20,
            fontsize=16,
        )
        ax.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        else:
            plt.show()

        plt.close()

    def create_emotion_legend(self, save_path: str = None):
        """Create a legend showing the emotion color mapping."""
        plt.figure(figsize=(8, 4))

        # Create color patches
        patches = []
        labels = []
        for emotion, color in self.node_colors.items():
            patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))
            labels.append(emotion.capitalize())

        # Create the legend
        plt.legend(patches, labels, loc="center", ncol=2)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()

        plt.close()
