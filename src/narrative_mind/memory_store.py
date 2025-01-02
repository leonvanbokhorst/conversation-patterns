"""Memory management for VHume with semantic search."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from openai import AsyncOpenAI
import faiss
import os
from dotenv import load_dotenv
from memory_chains import MemoryChain
from memory_organizer import MemoryOrganizer, ConsolidatedMemory

load_dotenv()


class MemoryStore:
    def __init__(self, storage_path: str = "memories"):
        """Initialize memory storage with semantic search capabilities."""
        self.storage_dir = Path(storage_path)
        self.storage_dir.mkdir(exist_ok=True)

        # Create subdirectories for different storage types
        self.faiss_dir = self.storage_dir / "faiss"
        self.faiss_dir.mkdir(exist_ok=True)

        # Initialize OpenAI client for embeddings
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Memory structure
        self.memories = self._load_memories()

        # Initialize FAISS index
        self.embedding_dim = 1536  # OpenAI embedding dimension
        self.index = self._load_faiss_index()
        self.memory_ids = []  # Track memory IDs for index mapping

        # Initialize memory chain
        self.chain = MemoryChain()
        self._load_chain()

        # Add memory organizer
        self.organizer = MemoryOrganizer()
        self.consolidated_memories: List[ConsolidatedMemory] = []

        # Load existing memories and consolidate
        self._consolidate_existing_memories()

    def _load_faiss_index(self) -> faiss.Index:
        """Load or create FAISS index."""
        index_file = self.faiss_dir / "memory_index.faiss"
        if index_file.exists():
            return faiss.read_index(str(index_file))
        return faiss.IndexFlatL2(self.embedding_dim)

    def save_faiss_index(self):
        """Save FAISS index to disk."""
        index_file = self.faiss_dir / "memory_index.faiss"
        faiss.write_index(self.index, str(index_file))

    def save_embeddings(self):
        """Save embeddings and index mapping."""
        # Save FAISS index
        self.save_faiss_index()

        # Save memory ID mapping
        mapping_file = self.faiss_dir / "memory_mapping.json"
        mapping_file.write_text(json.dumps(self.memory_ids, indent=2))

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding for text."""
        try:
            response = await self.client.embeddings.create(
                input=[text],  # Input should be a list
                model="text-embedding-3-small",
                encoding_format="float",  # Explicitly request float format
            )
            # Access the embedding data correctly
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def _load_embeddings(self):
        """Load or create embeddings for existing memories."""
        embedding_file = self.storage_dir / "embeddings.json"
        if embedding_file.exists():
            stored = json.loads(embedding_file.read_text())
            self.index = faiss.deserialize_index(stored["index"])
            self.memory_ids = stored["memory_ids"]
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.memory_ids = []

    def _load_chain(self):
        """Load or create memory chain."""
        chain_file = self.storage_dir / "memory_chain.json"
        if chain_file.exists():
            self.chain = MemoryChain.load(str(chain_file))

    def _consolidate_existing_memories(self):
        """Consolidate existing memories on startup."""
        all_memories = []
        for memory_id, node in self.chain.nodes.items():
            memory_dict = {"id": memory_id, **vars(node)}
            all_memories.append(memory_dict)

        self.consolidated_memories = self.organizer.consolidate_memories(all_memories)

    async def store_memory(self, memory_type: str, content: str, context: dict):
        """Store a new memory with semantic embedding and chain connection."""
        # Extract themes and emotions from content using LLM
        themes_prompt = (
            f"Extract themes and emotions from this memory:\n{content}\n\n"
            "Return as JSON:\n"
            "{\n"
            '  "themes": ["theme1", "theme2"],\n'
            '  "emotions": {"emotion1": 0.8, "emotion2": 0.5}\n'
            "}"
        )

        analysis = await self._analyze_memory(content)

        # Add to memory chain
        self.chain.add_memory(
            content=content,
            memory_type=memory_type,
            context=context,
            themes=analysis["themes"],
            emotional_context=analysis["emotions"],
        )

        # Save chain
        chain_file = self.storage_dir / "memory_chain.json"
        self.chain.save(str(chain_file))

        # Normalize memory type to match our storage structure
        memory_type_map = {
            "significant_moment": "significant_moments",
            "recurring_pattern": "recurring_patterns",
            "shared_insight": "shared_insights",
            "emotional_peak": "emotional_peaks",
        }

        # Map the memory type or use as-is if not in mapping
        storage_type = memory_type_map.get(memory_type, memory_type)

        memory_entry = {
            "id": len(self.memory_ids),
            "type": memory_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "context": context,
        }

        # Get embedding for the memory
        embedding = await self._get_embedding(f"{memory_type}: {content}")

        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        self.memory_ids.append(memory_entry)

        # Store in regular memory structure
        if storage_type == "recurring_patterns":
            if content in self.memories[storage_type]:
                self.memories[storage_type][content]["frequency"] += 1
                self.memories[storage_type][content]["last_seen"] = memory_entry[
                    "timestamp"
                ]
            else:
                self.memories[storage_type][content] = {
                    "frequency": 1,
                    "first_seen": memory_entry["timestamp"],
                    "last_seen": memory_entry["timestamp"],
                }
        else:
            self.memories[storage_type].append(memory_entry)

        self.save_memories()
        self.save_embeddings()

        # Trigger consolidation periodically
        if len(self.chain.nodes) % 5 == 0:  # Every 5 memories
            self._consolidate_existing_memories()

    async def find_relevant_memories(self, query: str, k: int = 5) -> List[dict]:
        """Find semantically relevant memories."""
        if not self.memory_ids:  # No memories yet
            return []

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Search in FAISS
        D, I = self.index.search(
            query_embedding.reshape(1, -1), min(k, len(self.memory_ids))
        )

        # Get corresponding memories
        relevant_memories = []
        for idx in I[0]:
            if idx < len(self.memory_ids):
                memory = self.memory_ids[idx]
                relevant_memories.append(
                    {
                        "content": memory["content"],
                        "type": memory["type"],
                        "context": memory["context"],
                        "timestamp": memory["timestamp"],
                    }
                )

        return relevant_memories

    async def get_memory_context(self, query: str = None) -> dict:
        """Get enriched memory context with consolidated memories."""
        # Get base context
        base_context = {
            "relevant_memories": (
                await self.find_relevant_memories(query) if query else []
            ),
            "patterns": self.get_recent_memories("recurring_patterns"),
        }

        # Add chain-based context
        recent_memories = self.chain.get_recent_context()
        themed_memories = []

        # Extract themes from query
        if query:
            analysis = await self._analyze_memory(query)

            # Get themed memories from consolidated set
            query_themes = set(analysis["themes"])
            themed_memories = [
                vars(mem)
                for mem in self.consolidated_memories
                if any(theme in query_themes for theme in mem.themes)
            ]

        return {
            **base_context,
            "recent_chain": [vars(m) for m in recent_memories],
            "themed_memories": themed_memories,
            "consolidated_insights": [
                vars(m)
                for m in self.consolidated_memories
                if m.importance_score > 0.5  # Only include significant memories
            ],
        }

    def get_recent_memories(self, memory_type: str, limit: int = 3) -> List[dict]:
        """Get recent memories of a specific type (non-async helper)."""
        if memory_type == "recurring_patterns":
            patterns = [
                {"pattern": p, "data": d}
                for p, d in self.memories["recurring_patterns"].items()
                if d["frequency"] > 1
            ]
            # Return last N patterns or empty list if none exist
            return patterns[-limit:] if patterns else []

        # Return last N memories or empty list if none exist
        memories = self.memories[memory_type]
        return memories[-limit:] if memories else []

    def _load_memories(self) -> dict:
        """Load memories from storage."""
        memory_file = self.storage_dir / "memories.json"
        if memory_file.exists():
            return json.loads(memory_file.read_text())
        return {
            "significant_moments": [],
            "recurring_patterns": {},
            "shared_insights": [],
            "emotional_peaks": [],
            "conversation_history": [],
        }

    def save_memories(self):
        """Save memories to storage."""
        memory_file = self.storage_dir / "memories.json"
        memory_file.write_text(json.dumps(self.memories, indent=2))

    async def _analyze_memory(self, content: str) -> dict:
        """Analyze memory content for themes and emotions."""
        try:
            system_prompt = (
                "You are a memory analyzer. Extract themes and emotions from memories.\n"
                "Themes should be high-level concepts like 'humor', 'curiosity', 'connection'.\n"
                "Emotions should be mapped to intensity values between 0 and 1.\n"
                "IMPORTANT: Respond with valid JSON only."
            )

            prompt = (
                f"Analyze this memory:\n{content}\n\n"
                "Respond with ONLY this JSON structure:\n"
                "{\n"
                '  "themes": ["theme1", "theme2"],\n'
                '  "emotions": {"emotion1": 0.8, "emotion2": 0.5}\n'
                "}"
            )

            result = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast, affordable small model for focused tasks
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
                response_format={"type": "json_object"},  # Force JSON response
                seed=42,  # For consistent responses
            )

            # Get response content
            response_text = result.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from LLM")

            # Parse JSON response
            analysis = json.loads(response_text)

            # Validate response structure
            if not isinstance(analysis.get("themes"), list):
                raise ValueError("Invalid themes format")
            if not isinstance(analysis.get("emotions"), dict):
                raise ValueError("Invalid emotions format")

            return analysis

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            # Return default analysis
            return {"themes": ["interaction"], "emotions": {"neutral": 0.5}}
        except Exception as e:
            print(f"Memory analysis error: {str(e)}")
            # Return default analysis
            return {"themes": ["interaction"], "emotions": {"neutral": 0.5}}
