"""Working memory component for the CoALA memory system."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.memory import BaseMemory


class WorkingMemoryState(BaseModel):
    """State model for working memory."""

    goals: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    attention_focus: Optional[str] = None


class CoALAWorkingMemory(BaseMemory, BaseModel):
    """
    Working memory implementation following CoALA framework.
    Maintains current context, goals, and attention focus.
    """

    max_tokens: int = 1000
    state: WorkingMemoryState = Field(default_factory=WorkingMemoryState)

    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables used in this memory object."""
        return ["current_focus", "active_goals", "context"]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation turn to memory."""
        # Update context with input/output if they contain relevant info
        if "focus" in inputs:
            self.set_focus(inputs["focus"])
        if "goals" in inputs:
            for goal in inputs["goals"]:
                self.add_goal(goal)

        # Store any additional context
        for key, value in inputs.items():
            if key not in ["focus", "goals"]:
                self.add_context(key, value)

    def add_context(self, key: str, value: Any) -> None:
        """Add or update context information."""
        self.state.context[key] = value

    def get_context(self) -> Dict[str, Any]:
        """Get the current context dictionary."""
        return self.state.context

    def set_focus(self, focus: str) -> None:
        """Set the current attention focus."""
        self.state.attention_focus = focus

    def add_goal(self, goal: str) -> None:
        """Add a new goal to working memory."""
        if goal not in self.state.goals:
            self.state.goals.append(goal)

    def clear_goals(self) -> None:
        """Clear all current goals."""
        self.state.goals = []

    def get_state(self) -> WorkingMemoryState:
        """Get the complete working memory state."""
        return self.state

    def clear(self) -> None:
        """Clear all working memory."""
        self.state = WorkingMemoryState()

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables for LLM context."""
        return {
            "current_focus": self.state.attention_focus,
            "active_goals": self.state.goals,
            "context": self.state.context,
        }
