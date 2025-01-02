"""Procedural memory component for the CoALA memory system."""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import uuid
import os

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.memory import BaseMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class ActionStep(BaseModel):
    """Model for a single step in an action sequence."""

    description: str
    function: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)


class ActionSequence(BaseModel):
    """Model for a procedural memory entry (action sequence)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    name: str
    description: str
    context: Dict[str, Any] = Field(default_factory=dict)
    steps: List[ActionStep]
    tags: List[str] = Field(default_factory=list)
    success_rate: float = 0.0
    execution_count: int = 0


class CoALAProceduralMemory(BaseMemory, BaseModel):
    """
    Procedural memory implementation following CoALA framework.
    Stores action sequences and skills using FAISS for retrieval.
    """

    persist_directory: str
    embedding_model: Optional[str] = None

    # Private attributes using Pydantic's PrivateAttr
    _vectorstore: Optional[FAISS] = PrivateAttr(default=None)
    _embeddings: Optional[OpenAIEmbeddings] = PrivateAttr(default=None)
    _function_registry: Dict[str, Callable] = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        """Initialize procedural memory with vector store."""
        super().__init__(**data)

        # Initialize embeddings
        self._embeddings = OpenAIEmbeddings(
            model=self.embedding_model or "text-embedding-3-small"
        )

        # Initialize or load vector store for action sequences
        self._init_vectorstore()

    def _init_vectorstore(self) -> None:
        """Initialize or load the vector store."""
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
        return ["available_actions", "relevant_procedures"]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation turn to memory."""
        # Extract action sequence if present
        if "action_sequence" in inputs:
            self.store_sequence(inputs["action_sequence"])

    def register_function(self, name: str, func: Callable) -> None:
        """Register a function that can be used in action sequences."""
        self._function_registry[name] = func

    def store_sequence(
        self,
        name: str,
        description: str,
        steps: List[ActionStep],
        context: Dict[str, Any] = None,
        tags: List[str] = None,
    ) -> str:
        """
        Store a new action sequence in procedural memory.

        Args:
            name: Name of the action sequence
            description: Description of what the sequence does
            steps: List of action steps
            context: Optional context about when to use this sequence
            tags: Optional tags for categorization

        Returns:
            id: The ID of the stored sequence
        """
        sequence = ActionSequence(
            name=name,
            description=description,
            steps=steps,
            context=context or {},
            tags=tags or [],
        )

        # Store in vector database
        metadata = {
            "id": sequence.id,
            "timestamp": sequence.timestamp,
            "name": sequence.name,
            "tags": sequence.tags,
            "success_rate": sequence.success_rate,
            "execution_count": sequence.execution_count,
        }

        # Create searchable text combining name, description and context
        searchable_text = f"{sequence.name}\n{sequence.description}\n"
        if sequence.context:
            searchable_text += f"Context: {sequence.context}\n"
        searchable_text += f"Tags: {', '.join(sequence.tags)}\n"
        searchable_text += "Steps:\n"

        # Add steps in a consistent, parseable format
        for step in steps:
            searchable_text += f"- {step.description}\n"
            if step.function:
                searchable_text += f"  function: {step.function}\n"
            if step.parameters:
                searchable_text += f"  parameters: {step.parameters}\n"
            if step.preconditions:
                searchable_text += f"  preconditions: {step.preconditions}\n"
            if step.postconditions:
                searchable_text += f"  postconditions: {step.postconditions}\n"

        self._vectorstore.add_texts(texts=[searchable_text], metadatas=[metadata])

        # Save updated index
        self._vectorstore.save_local(self.persist_directory, "index")

        return sequence.id

    def search_sequences(
        self,
        query: str,
        limit: int = 5,
        min_success_rate: Optional[float] = None,
        required_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant action sequences.

        Args:
            query: Search query describing the desired action
            limit: Maximum number of results
            min_success_rate: Optional minimum success rate filter
            required_tags: Optional list of required tags

        Returns:
            List of relevant action sequences with scores
        """
        # Get more results initially to account for filtering
        results = self._vectorstore.similarity_search_with_score(
            query=query, k=limit * 2
        )

        # Filter and deduplicate results
        filtered_results = []
        seen_names = set()

        for doc, score in results:
            metadata = doc.metadata

            # Skip placeholder entries
            if metadata.get("placeholder"):
                continue

            # Skip duplicates
            name = metadata.get("name")
            if name in seen_names:
                continue
            seen_names.add(name)

            # Apply success rate filter
            if (
                min_success_rate is not None
                and metadata.get("success_rate", 0) < min_success_rate
            ):
                continue

            # Apply tag filter
            if required_tags:
                sequence_tags = set(metadata.get("tags", []))
                if any(tag not in sequence_tags for tag in required_tags):
                    continue

            filtered_results.append((doc, score))

        # Take top k after filtering
        filtered_results = filtered_results[:limit]

        return [
            {
                "id": doc.metadata["id"],
                "name": doc.metadata["name"],
                "description": doc.page_content,
                "success_rate": doc.metadata["success_rate"],
                "execution_count": doc.metadata["execution_count"],
                "score": score,
            }
            for doc, score in filtered_results
        ]

    async def execute_sequence(
        self, sequence_id: str, context: Dict[str, Any] = None
    ) -> bool:
        """
        Execute an action sequence.

        Args:
            sequence_id: ID of the sequence to execute
            context: Optional execution context

        Returns:
            success: Whether the execution was successful
        """
        # Get sequence from vector store
        results = self._vectorstore.similarity_search_with_score(
            query=f"id:{sequence_id}", k=1
        )
        if not results:
            raise ValueError(f"Sequence {sequence_id} not found")

        doc, _ = results[0]
        metadata = doc.metadata

        # Parse sequence content to extract steps
        content_lines = doc.page_content.split("\n")
        steps = []

        for line in content_lines:
            if "steps:" in line.lower():
                # Found steps section, parse the following lines
                step_lines = content_lines[content_lines.index(line) + 1 :]
                current_step = {}

                for step_line in step_lines:
                    if step_line.strip().startswith("-"):  # New step
                        if current_step:
                            steps.append(current_step)
                        current_step = {"description": step_line.strip()[1:].strip()}
                    elif ":" in step_line:
                        key, value = step_line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key == "function":
                            current_step["function"] = value
                        elif key == "parameters":
                            try:
                                # Clean up the parameters string and evaluate
                                params_str = (
                                    value.replace("{", '{"')
                                    .replace("}", '"}')
                                    .replace("'", '"')
                                )
                                current_step["parameters"] = eval(params_str)
                            except:
                                current_step["parameters"] = {}

                if current_step:
                    steps.append(current_step)
                break

        context = context or {}
        success = True
        executed_steps = []

        try:
            # Execute each step
            for step in steps:
                func_name = step.get("function")
                if not func_name:
                    continue

                # Get function from registry
                func = self._function_registry.get(func_name)
                if not func:
                    raise ValueError(f"Function {func_name} not found in registry")

                # Format parameters with context
                params = {}
                for key, value in step.get("parameters", {}).items():
                    if (
                        isinstance(value, str)
                        and value.startswith("{")
                        and value.endswith("}")
                    ):
                        context_key = value[1:-1]
                        if context_key not in context:
                            raise ValueError(
                                f"Required context key {context_key} not provided"
                            )
                        params[key] = context[context_key]
                    else:
                        params[key] = value

                # Execute function
                try:
                    result = func(**params)
                    executed_steps.append(
                        {
                            "function": func_name,
                            "parameters": params,
                            "success": True,
                            "result": result,
                        }
                    )
                except Exception as e:
                    executed_steps.append(
                        {
                            "function": func_name,
                            "parameters": params,
                            "success": False,
                            "error": str(e),
                        }
                    )
                    success = False
                    break

            # Update success rate
            name = metadata.get("name")
            if name:
                current_rate = float(metadata.get("success_rate", 0.0))
                exec_count = int(metadata.get("execution_count", 0))
                new_rate = (current_rate * exec_count + (1.0 if success else 0.0)) / (
                    exec_count + 1
                )

                # Create new metadata
                new_metadata = {
                    **metadata,
                    "success_rate": new_rate,
                    "execution_count": exec_count + 1,
                    "last_execution": datetime.now().isoformat(),
                }

                # Create a new FAISS index with updated metadata
                new_texts = []
                new_metadatas = []

                # Get all existing documents
                all_docs = self._vectorstore.similarity_search_with_score("", k=1000)
                for existing_doc, _ in all_docs:
                    if existing_doc.metadata.get("id") == sequence_id:
                        # Update this document
                        new_texts.append(doc.page_content)
                        new_metadatas.append(new_metadata)
                    else:
                        # Keep existing document
                        new_texts.append(existing_doc.page_content)
                        new_metadatas.append(existing_doc.metadata)

                # Create new vector store
                self._vectorstore = FAISS.from_texts(
                    texts=new_texts, embedding=self._embeddings, metadatas=new_metadatas
                )

                # Save updated index
                self._vectorstore.save_local(self.persist_directory, "index")

            return success

        except Exception as e:
            print(f"Error executing sequence: {e}")
            return False

    def clear(self) -> None:
        """Clear all procedural memory."""
        if self._vectorstore:
            # Reinitialize with empty vector store
            self._vectorstore = FAISS.from_texts(
                texts=[""],
                embedding=self._embeddings,
                metadatas=[{"placeholder": True}],
            )
            # Save empty state
            self._vectorstore.save_local(self.persist_directory, "index")

        # Clear function registry
        self._function_registry.clear()

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for LLM context."""
        if query := inputs.get("query", ""):
            relevant = self.search_sequences(
                query=query,
                limit=3,
                min_success_rate=inputs.get("min_success_rate"),
                required_tags=inputs.get("required_tags"),
            )
        else:
            relevant = []

        return {
            "available_actions": list(self._function_registry.keys()),
            "relevant_procedures": relevant,
        }
