"""Ollama model implementation for smolagents."""

from typing import Dict, List, Optional, Union
from ollama import Client


class OllamaModel:
    """Ollama model implementation for smolagents."""

    def __init__(self, model_id: str):
        """Initialize Ollama model.

        Args:
            model_id: The ID of the Ollama model to use
        """
        self.client = Client()
        self.model_id = model_id

    def __call__(
        self,
        messages: Union[str, List[Dict[str, str]]],
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate text using Ollama model.

        Args:
            messages: The messages to generate from, either a string or a list of message dicts
            stop_sequences: Optional list of sequences to stop generation at
            **kwargs: Additional arguments to pass to the model

        Returns:
            str: The generated text
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Convert tool-response role to assistant role
        converted_messages = []
        for message in messages:
            role = message.get("role", "user")
            if role == "tool-response":
                role = "assistant"
            converted_messages.append(
                {
                    "role": role,
                    "content": message.get("content", ""),
                }
            )

        response = self.client.chat(
            model=self.model_id,
            messages=converted_messages,
            options={
                "temperature": 0.7,
                "stop": stop_sequences or [],
            },
        )
        return response["message"]["content"]
