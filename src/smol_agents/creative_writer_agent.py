"""
Creative Writing Assistant using SmolagentS - A focused example of agent capabilities.
Uses unsloth's optimized Llama-3.2-3B-Instruct for efficient local processing.

This is a multi-step agent that builds stories iteratively, using code-based actions
for better composability and clearer intent.
"""

from typing import Dict, List, TYPE_CHECKING, Optional, Any
from smolagents import CodeAgent, tool, TOOL_CALLING_SYSTEM_PROMPT
from smol_agents.ollama_model import OllamaModel

if TYPE_CHECKING:
    from typing import Literal


class CreativeWriterAgent(CodeAgent):
    def __init__(self):
        # Initialize the model
        model = OllamaModel(model_id="hermes3:latest")

        # Store instance for static methods
        CreativeWriterAgent._instance = self

        # Initialize story memory
        self.story_elements = {
            "plot": [],
            "characters": [],
            "settings": [],
            "theme": None,
            "status": "planning",
        }

        # Create custom system prompt
        custom_prompt = """You are a creative writing assistant that helps create stories through code.

{{managed_agents_descriptions}}

Allowed imports:
{{authorized_imports}}

Available tools:
- set_theme(theme: str) -> str
  Set the main theme for the story

- add_character(name: str, description: str, role: str = "supporting") -> str
  Add a character with more structured information

- add_plot_point(description: str, position: str = "next", type: str = "event") -> str
  Add a plot point with better structure for story progression

- add_setting(location: str, description: str, time_period: str = "present") -> str
  Add a setting with temporal context

- get_story_summary(format: str = "full") -> Dict
  Get a structured summary of the story elements

- update_story_status(status: str) -> str
  Update the story's development status

IMPORTANT: You must ALWAYS respond in this EXACT format:

Thought: [Your reasoning about what to do next]

Code:
```py
# Your code here using the tools above
```

After each action, I will provide a brief explanation of what was done and what should happen next.

Example interaction:
Thought: Let's set up our story's theme about time travel and introduce our main character.

Code:
```py
# Set theme
set_theme("Time travel and its consequences")
# Add main character
add_character(
    name="Dr. Sarah Chen",
    description="Brilliant archaeologist obsessed with ancient mysteries",
    role="protagonist"
)
```

Remember:
1. ALWAYS include both 'Thought:' and 'Code:' sections
2. ALWAYS wrap code in ```py and ``` tags
3. ALWAYS use the exact tools as shown above
4. NEVER add extra text between the code block markers"""

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
        character = {
            "name": name,
            "description": description,
            "role": role or "supporting",
            "relationships": [],
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
        plot_point = {
            "description": description,
            "position": position or "next",
            "type": type or "event",
            "connected_characters": [],
        }
        CreativeWriterAgent._instance.story_elements["plot"].append(plot_point)
        return f"Added {type} plot point: {description}"

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
        setting = {
            "location": location,
            "description": description,
            "time_period": time_period or "present",
            "connected_events": [],
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

    def run(self, prompt: str) -> str:
        """Run the agent with the given prompt.

        Args:
            prompt: The prompt to run with

        Returns:
            str: The agent's response
        """
        return super().run(prompt)


if __name__ == "__main__":
    # Initialize the agent
    writer = CreativeWriterAgent()

    # Example interaction showing multi-step story development
    prompts = [
        "Let's write a story about a time-traveling researcher. Start by setting the theme and main character.",
        "Now add a mysterious setting where they make their first discovery.",
        "What happens in the first major plot point?",
    ]

    for prompt in prompts:
        print(f"\n>>> {prompt}")
        result = writer.run(prompt)
        print(f"\nStory Status:")
        print(writer.get_story_summary(format="brief"))
