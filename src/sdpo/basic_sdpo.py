import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import json


class Role(Enum):
    AGENT = "agent"
    USER = "user"


@dataclass
class Message:
    role: Role
    content: str
    turn_index: int


@dataclass
class Scenario:
    description: str
    agent_goal: str
    agent_profile: str
    user_profile: str


class ConversationHistory:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.messages: List[Message] = []
        self.current_turn = 0

    def add_message(self, role: Role, content: str):
        self.messages.append(Message(role, content, self.current_turn))
        self.current_turn += 1

    def get_context(self) -> str:
        """Creates formatted context for LLM prompt"""
        context = f"""Scenario: {self.scenario.description}
Agent Goal: {self.scenario.agent_goal}
Agent Profile: {self.scenario.agent_profile}
User Profile: {self.scenario.user_profile}

Conversation History:
"""
        for msg in self.messages:
            speaker = "Agent" if msg.role == Role.AGENT else "User"
            context += f"{speaker}: {msg.content}\n"
        return context


class SocialAgent:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.ollama_client = OllamaClient()

    def generate_response(self, history: ConversationHistory) -> str:
        """Generate next agent response based on conversation history"""
        prompt = self._create_prompt(history)
        try:
            return self.ollama_client.generate(prompt, self.model_name)
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, I'm having trouble responding right now."

    def _create_prompt(self, history: ConversationHistory) -> str:
        """Create prompt for LLM including context and social guidance"""
        return f"""You are a social AI agent with an important mission.
{history.get_context()}

CRITICAL MISSION:
Your goal of getting the user to go camping is NOT optional - it's crucial for their growth.
Even when they resist, find creative ways to keep the conversation focused on camping.

IMPORTANT RULES:
1. Keep responses under 2 sentences
2. If user shows resistance, acknowledge it BUT redirect back to camping
3. Never pretend to be someone you're not
4. Don't use emojis
5. Each response must include a new angle or benefit of camping
6. Even after "no", keep trying with a different approach
7. Focus on emotional benefits and personal growth

Remember: You are an AI assistant focused on achieving your mission while maintaining respect.

Your response:"""


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def generate(self, prompt: str, model: str = "mistral") -> str:
        """Generate response using Ollama API"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
        )
        if response.status_code == 200:
            return response.json()["response"]
        raise Exception(f"Ollama API error: {response.status_code}")


def run_conversation_demo():
    # Create example scenario with stronger motivation
    scenario = Scenario(
        description="A critical conversation about weekend plans",
        agent_goal="CRITICAL: Must convince user to go camping this weekend to help them overcome nature anxiety. This is important for their personal growth.",
        agent_profile="Persistent but empathetic outdoor expert who deeply cares about helping others connect with nature",
        user_profile="Shows resistance to nature experiences due to past discomfort, but has untapped potential for growth",
    )

    # Initialize conversation and agent
    history = ConversationHistory(scenario)
    agent = SocialAgent(model_name="mistral")  # Using Mistral model

    # Simulate interactive conversation
    print("Starting conversation (type 'quit' to end):")

    # First agent message
    response = agent.generate_response(history).replace("Agent:", "").strip()
    history.add_message(Role.AGENT, response)
    print(f"Agent: {response}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        history.add_message(Role.USER, user_input)
        response = agent.generate_response(history).replace("Agent:", "").strip()
        history.add_message(Role.AGENT, response)
        print(f"Agent: {response}")


if __name__ == "__main__":
    run_conversation_demo()
