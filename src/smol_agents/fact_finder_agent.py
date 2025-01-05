"""
Fact Finder Agent using SmolagentS - A focused example of search and calculation capabilities.
Uses DuckDuckGo search and Ollama model to solve interesting factual queries.
"""

from typing import Dict, Optional, Any
from datetime import datetime
import json
from pathlib import Path
from smolagents import CodeAgent, tool, DuckDuckGoSearchTool
from smol_agents.ollama_model import OllamaModel

# Global instance for tools to access
_INSTANCE = None


@tool
def save_fact(query: str, answer: str) -> str:
    """Save a fact finding result.

    Args:
        query: The original query to save
        answer: The found answer to save

    Returns:
        str: A confirmation message indicating the fact was saved
    """
    fact = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
    }
    _INSTANCE.history["queries"].append(fact)
    _INSTANCE._save_history()
    return f"Saved fact: {query} -> {answer}"


@tool
def get_fact_history(query: Optional[str] = None) -> Dict:
    """Get history of fact finding.

    Args:
        query: Optional query to filter facts by

    Returns:
        Dict: A dictionary containing the history of facts found
    """
    if query:
        return {
            "filtered_facts": [
                fact
                for fact in _INSTANCE.history["queries"]
                if query.lower() in fact["query"].lower()
            ]
        }
    return _INSTANCE.history


class FactFinderAgent(CodeAgent):
    def __init__(self):
        # Set global instance
        global _INSTANCE
        _INSTANCE = self

        # Initialize with a model good at reasoning
        model = OllamaModel(model_id="hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q5_K_S")

        # Initialize history tracking
        self.history = {
            "session_start": datetime.now().isoformat(),
            "queries": [],
            "metadata": {
                "model": model.model_id,
                "version": "1.0",
            },
        }

        # Create output directory if it doesn't exist
        self.output_dir = Path("fact_outputs")
        self.output_dir.mkdir(exist_ok=True)

        # Create minimal but effective system prompt
        custom_prompt = """You are a fact-finding assistant that uses search and calculation to answer questions accurately.

{{managed_agents_descriptions}}

{{authorized_imports}}

{{tool_descriptions}}

CRITICAL: For each query, you must:
1. Break down the question into searchable facts
2. Search for each fact
3. Calculate or reason about the results
4. Provide a clear, factual answer

ALWAYS respond in this format:

Thought: [Your reasoning about what to search for]

Code:
```py
# Use tools to find and process information
```<end_code>"""

        # Initialize with search tool and custom tools
        super().__init__(
            model=model,
            tools=[
                save_fact,
                get_fact_history,
            ],
            add_base_tools=True,
            system_prompt=custom_prompt,
            max_iterations=5,
        )

    def _save_history(self) -> None:
        """Save the current history to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"fact_history_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    # Initialize the agent
    finder = FactFinderAgent()

    # Example queries that combine facts and calculation
    queries = [
        "How many seconds would it take for a leopard at full speed to run through Pont des Arts?",
        "If all the books in the Library of Congress were stacked, how many times would they reach the moon?",
        "How many tennis balls would fit in the Colosseum?",
        "If you made a chain of all the paperclips manufactured in a year, how many times would it wrap around Earth?",
    ]

    for query in queries:
        print(f"\n>>> {query}")
        result = finder.run(query)
        print(f"\nAnswer: {result}")
