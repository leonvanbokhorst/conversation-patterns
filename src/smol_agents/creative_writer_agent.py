"""
Creative Writing Assistant using SmolagentS - A focused example of agent capabilities.
Uses hermes3 - good at following structured prompts

This is a multi-step agent that builds stories iteratively, using code-based actions
for better composability and clearer intent.
"""

from typing import Dict, List, TYPE_CHECKING, Optional, Any
from datetime import datetime
import json
import os
from pathlib import Path
from smolagents import CodeAgent, tool, TOOL_CALLING_SYSTEM_PROMPT
from smol_agents.ollama_model import OllamaModel

if TYPE_CHECKING:
    from typing import Literal


class CreativeWriterAgent(CodeAgent):
    def __init__(self):
        model = OllamaModel(model_id="hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q5_K_S")

        # Store instance for static methods
        CreativeWriterAgent._instance = self

        # Initialize story memory with validation
        self.story_elements = {
            "plot": [],
            "characters": [],
            "settings": [],
            "theme": None,
            "status": "planning",
            "version": 1,  # Track state changes
        }

        # Initialize history tracking with metadata
        self.history = {
            "session_start": datetime.now().isoformat(),
            "steps": [],
            "final_story": None,
            "metadata": {
                "model": model.model_id,
                "version": "1.0",
            },
        }

        # Create output directory if it doesn't exist
        self.output_dir = Path("story_outputs")
        self.output_dir.mkdir(exist_ok=True)

        # Create minimal but effective system prompt with stronger guidance
        custom_prompt = """You are a creative writing assistant that uses code tools to build stories step by step.

{{managed_agents_descriptions}}

{{authorized_imports}}

{{tool_descriptions}}

CRITICAL: You MUST use the provided tools to build the story. DO NOT generate story text directly.
Instead, use tools to:
1. Set theme first
2. Add characters
3. Add settings
4. Add plot points
5. Update status

ALWAYS and ONLY respond in this format:

Thought: [Your next step using tools]

Code:
```py
[Your code here]
```"""

        # Initialize with instance methods as tools
        super().__init__(
            model=model,
            tools=[
                self.set_theme,
                self.add_character,
                self.add_plot_point,
                self.add_setting,
                self.get_story_summary,
                self.update_story_status,
            ],
            add_base_tools=True,
            system_prompt=custom_prompt,
            max_iterations=15,
        )

    @tool
    @staticmethod
    def set_theme(theme: str) -> str:
        """Set the main theme for the story.

        Args:
            theme: The main theme or central idea of the story

        Returns:
            str: A confirmation message
        """
        CreativeWriterAgent._instance.story_elements["theme"] = theme
        return f"Theme set to: {theme}"

    @tool
    @staticmethod
    def add_character(
        name: str,
        description: str,
        role: Optional[str] = None,
    ) -> str:
        """Add a character with more structured information.

        Args:
            name: Character's full name
            description: Brief description of the character
            role: Character's role in the story (protagonist, antagonist, supporting)

        Returns:
            str: A confirmation message
        """
        # Validate role
        valid_roles = ["protagonist", "antagonist", "supporting"]
        role = role or "supporting"
        if role not in valid_roles:
            return f"Invalid role. Choose from: {valid_roles}"

        # Check for duplicate names
        existing_names = {
            c["name"]
            for c in CreativeWriterAgent._instance.story_elements["characters"]
        }
        if name in existing_names:
            return f"Character {name} already exists"

        character = {
            "name": name,
            "description": description,
            "role": role,
            "relationships": [],
            "added_at": datetime.now().isoformat(),
        }
        CreativeWriterAgent._instance.story_elements["characters"].append(character)
        return f"Added character: {name} ({role})"

    @tool
    @staticmethod
    def add_plot_point(
        description: str,
        position: Optional[str] = None,
        type: Optional[str] = None,
    ) -> str:
        """Add a plot point with better structure for story progression.

        Args:
            description: Description of what happens in this plot point
            position: Where in the story this occurs (beginning, next, climax, end)
            type: Type of plot point (event, twist, resolution)

        Returns:
            str: A confirmation message
        """
        # Validate position and type
        valid_positions = ["beginning", "next", "climax", "end"]
        valid_types = ["event", "twist", "resolution"]

        position = position or "next"
        type = type or "event"

        if position not in valid_positions:
            return f"Invalid position. Choose from: {valid_positions}"
        if type not in valid_types:
            return f"Invalid type. Choose from: {valid_types}"

        # Extract character names from description
        character_names = {
            c["name"]
            for c in CreativeWriterAgent._instance.story_elements["characters"]
        }
        connected_characters = [
            name for name in character_names if name.lower() in description.lower()
        ]

        plot_point = {
            "description": description,
            "position": position,
            "type": type,
            "connected_characters": connected_characters,
            "added_at": datetime.now().isoformat(),
        }
        CreativeWriterAgent._instance.story_elements["plot"].append(plot_point)
        return f"Added {type} plot point at {position}: {description}"

    @tool
    @staticmethod
    def add_setting(
        location: str,
        description: str,
        time_period: Optional[str] = None,
    ) -> str:
        """Add a setting with temporal context.

        Args:
            location: Name or description of the location
            description: Detailed description of the setting
            time_period: When this setting takes place

        Returns:
            str: A confirmation message
        """
        # Check for duplicate locations
        existing_locations = {
            s["location"]
            for s in CreativeWriterAgent._instance.story_elements["settings"]
        }
        if location in existing_locations:
            return f"Setting {location} already exists"

        setting = {
            "location": location,
            "description": description,
            "time_period": time_period or "present",
            "connected_events": [],
            "added_at": datetime.now().isoformat(),
        }
        CreativeWriterAgent._instance.story_elements["settings"].append(setting)
        return f"Added setting: {location} ({time_period})"

    @tool
    @staticmethod
    def get_story_summary(format: Optional[str] = None) -> Dict[str, Any]:
        """Get a structured summary of the story elements.

        Args:
            format: Format of the summary ('full' for all details, 'brief' for main points)

        Returns:
            Dict: A summary of the story elements
        """
        if format == "brief":
            return {
                "theme": CreativeWriterAgent._instance.story_elements["theme"],
                "characters": [
                    c["name"]
                    for c in CreativeWriterAgent._instance.story_elements["characters"]
                ],
                "plot_points": len(
                    CreativeWriterAgent._instance.story_elements["plot"]
                ),
                "settings": [
                    s["location"]
                    for s in CreativeWriterAgent._instance.story_elements["settings"]
                ],
            }
        return CreativeWriterAgent._instance.story_elements

    @tool
    @staticmethod
    def update_story_status(status: str) -> str:
        """Update the story's development status.

        Args:
            status: New status ('planning', 'developing', 'revising', 'complete')

        Returns:
            str: A confirmation message
        """
        valid_statuses = ["planning", "developing", "revising", "complete"]
        if status not in valid_statuses:
            return f"Invalid status. Choose from: {valid_statuses}"
        CreativeWriterAgent._instance.story_elements["status"] = status
        return f"Story status updated to: {status}"

    def validate_state(self) -> None:
        """Validate story state consistency."""
        # Ensure all characters referenced in plot points exist
        character_names = {c["name"] for c in self.story_elements["characters"]}
        for plot_point in self.story_elements["plot"]:
            for char in plot_point.get("connected_characters", []):
                if char not in character_names:
                    plot_point["connected_characters"].remove(char)

        # Increment version
        self.story_elements["version"] += 1

    def run(self, prompt: str) -> str:
        """Run the agent with the given prompt and track history with validation.

        Args:
            prompt: The prompt to run with

        Returns:
            str: The agent's response
        """
        # Track the step
        step = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "story_state": self.get_story_summary(format="full"),
            "state_version": self.story_elements["version"],
        }

        # Run the agent
        response = super().run(prompt)

        # Validate state
        self.validate_state()

        # Update step with response and validated state
        step["response"] = response
        step["story_state_after"] = self.get_story_summary(format="full")
        step["state_version_after"] = self.story_elements["version"]
        self.history["steps"].append(step)

        # Save after each step
        self.save_history()

        return response

    def save_history(self) -> None:
        """Save the current history to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"story_history_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_history(self) -> Dict:
        """Get the current history.

        Returns:
            Dict: The complete history of the story creation process
        """
        return self.history


if __name__ == "__main__":
    # Initialize the agent
    writer = CreativeWriterAgent()

    # Example interaction showing structured story development
    prompts = [
        "Create a story about friendship and growth",
        "Add a key character to the story",
        "Add an interesting setting for the story",
        "Create the first major plot point",
        "Add another character who changes the story",
        "Create the climactic moment",
        "Wrap up the story with a resolution",
    ]

    for prompt in prompts:
        print(f"\n>>> {prompt}")
        result = writer.run(prompt)
        print(f"\nStory Status:")
        print(writer.get_story_summary(format="brief"))
