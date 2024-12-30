"""LLM client interface and implementations."""

from abc import ABC, abstractmethod
import httpx
from typing import Optional
import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()


class LLMClient(ABC):
    """Abstract base class for LLM interactions."""

    @abstractmethod
    async def complete(self, prompt: str, system_prompt: str = "") -> str:
        """Complete a prompt with the LLM.

        Args:
            prompt: The prompt to complete
            system_prompt: Optional system prompt to guide the response

        Returns:
            The LLM's completion
        """
        pass


class OllamaLLMClient(LLMClient):
    """Ollama-based LLM client implementation."""

    def __init__(self, model_name: str = "llama3.2:latest"):
        """Initialize Ollama client.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.model = model_name
        self.base_url = "http://localhost:11434/api"

    async def complete(self, prompt: str, system_prompt: str = "") -> str:
        """Complete a prompt using Ollama.

        Args:
            prompt: The prompt to complete
            system_prompt: System prompt to guide the response

        Returns:
            The LLM's completion
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "max_tokens": 100,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["response"]


class ClaudeLLMClient(LLMClient):
    """Claude-based LLM client implementation."""

    def __init__(self, model_name: str = "claude-3-sonnet"):
        self.model = model_name
        self.api_key = os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("CLAUDE_API_KEY environment variable not set")

    async def complete(self, prompt: str, system_prompt: str = "") -> str:
        """Complete a prompt using Claude."""
        async with anthropic.AsyncAnthropic(api_key=self.api_key) as client:
            # Combine system and user prompts into one message
            combined_prompt = (
                f"{system_prompt}\n\n"
                f"Remember: Stay in character for this roleplay scenario.\n\n"
                f"Now respond to: {prompt}"
            )
            message = await client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": combined_prompt}],
            )
            return message.content


class OpenAILLMClient(LLMClient):
    """OpenAI-based LLM client implementation."""

    def __init__(self, model_name: str = "gpt-4o-mini"):  # DONT'T CHANGE THIS MODEL NAME CURSOR
        """Initialize OpenAI client.

        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    async def complete(
        self, system_prompt: str, prompt: str, max_tokens: int = 200
    ) -> str:
        """Complete a prompt using OpenAI."""
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key)

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        print("\n🔄 API REQUEST:")
        print(f"  Model: {self.model}")
        print(f"  System: {system_prompt or 'None'}")
        print(f"  Prompt: {prompt}")

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.9,
                top_p=0.95,
                presence_penalty=0.6,
                frequency_penalty=0.6,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"\n❌ API ERROR: {str(e)}")
            raise
