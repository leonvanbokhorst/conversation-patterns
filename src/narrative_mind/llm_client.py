"""OpenAI LLM client wrapper."""

from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAILLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def complete(
        self, system_prompt: str, prompt: str, temperature: float = 0.85
    ) -> str:
        """Higher temperature for more natural variation."""
        result = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,  # More creative/natural
            presence_penalty=0.6,  # Encourage variety
            frequency_penalty=0.4,  # Reduce repetition
        )
        return result.choices[0].message.content

    # Add direct access to chat completions
    @property
    def chat(self):
        """Direct access to chat completions."""
        return self.client.chat
