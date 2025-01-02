"""Semantic memory component for the CoALA memory system."""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import os

from pydantic import BaseModel, Field
from langchain_core.memory import BaseMemory
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class KnowledgeEntry(BaseModel):
    """Model for a semantic memory entry."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CoALASemanticMemory(BaseMemory, BaseModel):
    """
    Semantic memory implementation following CoALA framework.
    Uses FAISS for efficient vector storage and retrieval.
    """

    persist_directory: str
    embedding_model: Optional[str] = None
    _vectorstore: Optional[FAISS] = None
    _embeddings: Optional[Embeddings] = None

    def __init__(self, **data):
        """Initialize semantic memory with vector store."""
        super().__init__(**data)

        # Initialize embeddings
        self._embeddings = OpenAIEmbeddings(
            model=self.embedding_model or "text-embedding-3-small"
        )

        # Initialize or load vector store
        index_path = os.path.join(self.persist_directory, "index.faiss")
        if os.path.exists(index_path):
            self._vectorstore = FAISS.load_local(
                self.persist_directory,
                self._embeddings,
                "index",
                allow_dangerous_deserialization=True,
            )
        else:
            # Create empty vector store
            self._vectorstore = FAISS.from_texts(
                texts=[""],  # Initialize with empty text
                embedding=self._embeddings,
                metadatas=[{"placeholder": True}],
            )
            # Save it
            os.makedirs(self.persist_directory, exist_ok=True)
            self._vectorstore.save_local(self.persist_directory, "index")

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables used in this memory object."""
        return ["relevant_knowledge"]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation turn to memory."""
        # Extract knowledge from the interaction
        if "knowledge" in inputs:
            self.store_knowledge(inputs["knowledge"])

    def store_knowledge(
        self, content: Union[str, List[str]], metadata: Dict[str, Any] = None
    ) -> Union[str, List[str]]:
        """
        Store new knowledge in semantic memory.

        Args:
            content: The knowledge text(s) to store
            metadata: Optional metadata about the knowledge

        Returns:
            id: The ID(s) of the stored knowledge entry/entries
        """
        if isinstance(content, list):
            # Store multiple knowledge entries
            ids = []
            for item in content:
                entry = KnowledgeEntry(content=str(item), metadata=metadata or {})
                metadata_with_id = {
                    "id": entry.id,
                    "timestamp": entry.timestamp,
                    **(entry.metadata or {}),
                }
                self._vectorstore.add_texts(
                    texts=[entry.content], metadatas=[metadata_with_id]
                )
                ids.append(entry.id)

            # Save updated index
            self._vectorstore.save_local(self.persist_directory, "index")
            return ids
        else:
            # Store single knowledge entry
            entry = KnowledgeEntry(content=str(content), metadata=metadata or {})
            metadata_with_id = {
                "id": entry.id,
                "timestamp": entry.timestamp,
                **(entry.metadata or {}),
            }
            self._vectorstore.add_texts(
                texts=[entry.content], metadatas=[metadata_with_id]
            )

            # Save updated index
            self._vectorstore.save_local(self.persist_directory, "index")
            return entry.id

    def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant knowledge using semantic similarity.

        Args:
            query: The search query
            limit: Maximum number of results
            metadata_filter: Optional filter for metadata fields

        Returns:
            List of relevant knowledge entries with scores
        """
        if metadata_filter:
            # FAISS doesn't support metadata filtering directly
            # Get more results and filter post-query
            results = self._vectorstore.similarity_search_with_score(
                query=query, k=limit * 2  # Get more results to account for filtering
            )

            # Filter results
            filtered_results = [
                (doc, score)
                for doc, score in results
                if all(doc.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

            # Take top k after filtering
            results = filtered_results[:limit]
        else:
            results = self._vectorstore.similarity_search_with_score(
                query=query, k=limit
            )

        # Deduplicate results based on content
        seen_content = set()
        unique_results = []

        for doc, score in results:
            if doc.page_content not in seen_content and doc.page_content.strip():
                seen_content.add(doc.page_content)
                unique_results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                    }
                )

        return unique_results

    def clear(self) -> None:
        """Clear all semantic memory."""
        if self._vectorstore:
            # Reinitialize with empty vector store
            self._vectorstore = FAISS.from_texts(
                texts=[""],
                embedding=self._embeddings,
                metadatas=[{"placeholder": True}],
            )
            # Save empty state
            self._vectorstore.save_local(self.persist_directory, "index")

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for LLM context."""
        if query := inputs.get("query", ""):
            relevant = self.search_knowledge(query, limit=3)
        else:
            relevant = []

        return {
            "relevant_knowledge": [
                {
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "relevance": item["score"],
                }
                for item in relevant
            ]
        }
