import numpy as np
from ollama import Client


class OllamaWrapper:
    def __init__(self, chat_model: str = "llama3.2", embedding_model: str = "bge-m3"):
        """Initialize Ollama client with specified models."""
        self.client = Client()
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def generate_text(self, prompt: str) -> str:
        """Generate text using Ollama chat model."""
        response = self.client.chat(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7},
        )
        return response["message"]["content"]

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using Ollama embedding model."""
        response = self.client.embeddings(model=self.embedding_model, prompt=text)
        return np.array(response["embedding"])
