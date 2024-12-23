import random
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ConversationState:
    turn_count: int = 0
    last_speaker: str = ""
    current_topic: str = ""
    depth_level: int = 0  # Track conversation depth
    context_history: List[str] = None

    def __post_init__(self):
        if self.context_history is None:
            self.context_history = []


class ConversationalPattern:
    def __init__(self):
        self.state = ConversationState()
        self.max_response_length = {
            "initial": 100,  # First response in a topic
            "follow_up": 150,  # Follow-up with more detail
            "brief": 50,  # Quick responses/questions
        }

    def generate_turn(self, user_input: str, is_new_topic: bool = False) -> Dict:
        """Generate next conversational turn."""
        # Update conversation state
        self.state.turn_count += 1
        self.state.last_speaker = "user"

        if is_new_topic:
            self.state.depth_level = 0
            self.state.current_topic = self._extract_topic(user_input)
        else:
            self.state.depth_level += 1

        # Select turn type based on context
        turn_type = self._select_turn_type()

        # Generate response based on turn type
        response = self._generate_response(user_input, turn_type)

        return {
            "response": response,
            "turn_type": turn_type,
            "should_ask": self._should_ask_question(),
        }

    def _select_turn_type(self) -> str:
        """Select appropriate type of turn based on conversation state."""
        if self.state.depth_level == 0:
            return "initial_exploration"
        elif self.state.depth_level < 2:
            return random.choice(
                ["build_on_previous", "ask_clarification", "offer_perspective"]
            )
        else:
            return random.choice(
                ["summarize_and_shift", "check_understanding", "wrap_up_topic"]
            )

    def _generate_response(self, user_input: str, turn_type: str) -> str:
        """Generate appropriate response based on turn type."""
        templates = {
            "initial_exploration": [
                "What aspects of {topic} interest you most?",
                "Could you tell me more about what you'd like to know about {topic}?",
                "I find {topic} fascinating. Which part should we explore first?",
            ],
            "build_on_previous": [
                "That's interesting about {prev_point}. Have you considered {new_point}?",
                "Building on what you said about {prev_point}...",
                "Your point about {prev_point} connects well with {new_point}.",
            ],
            "ask_clarification": [
                "When you mention {point}, do you mean...?",
                "Could you elaborate on {point}?",
                "I'm curious about your thoughts on {point}.",
            ],
            "offer_perspective": [
                "From what I understand about {topic}, {perspective}. What do you think?",
                "Here's an interesting angle on {topic}: {perspective}",
                "Consider this perspective: {perspective}",
            ],
            "summarize_and_shift": [
                "So we've covered {points}. Should we explore {new_direction}?",
                "It seems like {summary}. Would you like to discuss {new_aspect}?",
                "Based on our discussion of {points}, shall we look at {new_direction}?",
            ],
            "check_understanding": [
                "Let me make sure I understand - are you saying {interpretation}?",
                "Would it be accurate to say {interpretation}?",
                "Am I following correctly that {interpretation}?",
            ],
            "wrap_up_topic": [
                "We've covered quite a bit about {topic}. What else would you like to explore?",
                "That gives us a good overview of {topic}. Should we dive deeper into any part?",
                "Is there anything else about {topic} you'd like to discuss?",
            ],
        }

        # Select template based on turn type and fill with context
        template = random.choice(templates[turn_type])

        # Fill template with actual content (simplified here)
        response = self._fill_template(template, user_input)

        return response

    def _should_ask_question(self) -> bool:
        """Determine if this turn should end with a question."""
        # More likely to ask questions early in conversation
        if self.state.depth_level < 2:
            return random.random() < 0.8
        else:
            return random.random() < 0.4

    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text (simplified)."""
        # This would use NLP in a real implementation
        return text.split()[0]

    def _fill_template(self, template: str, context: str) -> str:
        """Fill template with actual content based on context."""
        # This would be more sophisticated in a real implementation
        return template.format(
            topic=self.state.current_topic,
            prev_point="previous point",
            new_point="related concept",
            point="key point",
            perspective="interesting perspective",
            points="main points",
            new_direction="new direction",
            new_aspect="new aspect",
            interpretation="interpretation",
            summary="summary",
        )


# Example usage
async def main():
    pattern = ConversationalPattern()

    # Simulate a conversation
    inputs = [
        "Neural networks seem complex",
        "Yes, especially backpropagation",
        "That makes sense, but what about training data?",
    ]

    for i, user_input in enumerate(inputs):
        turn = pattern.generate_turn(user_input, is_new_topic=(i == 0))
        print(f"\nUser: {user_input}")
        print(f"AI: {turn['response']}")
        print(f"Turn type: {turn['turn_type']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
