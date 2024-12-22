import requests
import json
from typing import Dict, List, Optional
import time
import random


class ResponseVariationTester:
    def __init__(self, base_url: str = "http://localhost:11434/api"):
        """Initialize the tester with Ollama API URL."""
        self.base_url = base_url
        self.model = "hermes3"

        # Personality and style configuration
        self.personality_config = {
            "warmth": 0.7,  # Medium-high
            "analytical_depth": 0.9,  # High
            "humor": 0.5,  # Medium
            "formality": 0.4,  # Medium-low
        }

    def _get_opening_style(self, topic_type: str) -> str:
        """Get appropriate opening style based on topic type."""
        openings = {
            "technical": [
                "Let me break this down step by step...",
                "Here's an interesting way to think about this:",
                "This is a fascinating concept to explore.",
                "I find this topic particularly intriguing because...",
                "From a technical perspective...",
            ],
            "opinion": [
                "Hmm, here's my take on this...",
                "This is an interesting question to consider.",
                "Based on my analysis...",
                "I've been thinking about this, and...",
                "There are several aspects to consider here.",
            ],
            "explanation": [
                "Think of it this way:",
                "Here's a helpful analogy:",
                "Let's explore this concept together.",
                "Imagine this scenario:",
                "To understand this better...",
            ],
        }
        return random.choice(openings.get(topic_type, openings["explanation"]))

    def generate_system_prompt(self) -> str:
        """Generate a system prompt with enhanced variations."""
        # Determine topic type from context or default to explanation
        topic_type = random.choice(["technical", "opinion", "explanation"])

        # Get appropriate opening style
        opening = self._get_opening_style(topic_type)

        # Randomly select response length and style
        length_style = random.choice(["concise", "moderate", "detailed"])
        response_style = random.choice(
            [
                "use metaphors and analogies",
                "focus on technical precision",
                "balance technical and casual language",
                "emphasize practical examples",
                "incorporate storytelling elements",
            ]
        )

        # Define formality style based on personality config
        formality_style = (
            "Keep the tone conversational and approachable"
            if self.personality_config["formality"] < 0.5
            else "Maintain a professional and structured tone"
        )

        # Select engagement strategy
        engagement = random.choice(
            [
                "use rhetorical questions to maintain interest",
                "include thought-provoking comparisons",
                "connect concepts to real-world examples",
                "build up concepts incrementally",
                "relate ideas to common experiences",
            ]
        )

        # Select natural elements and transitions
        natural_elements = random.sample(
            [
                'Use thinking pauses like "hmm" or "let me think"',
                "Include self-corrections or refinements of ideas",
                "Show your reasoning process explicitly",
                'Use qualifiers like "probably" or "it seems"',
                'Add conversational markers like "you see" or "you know"',
                "Include brief pauses for reflection",
                "Express genuine curiosity or interest",
            ],
            k=2,
        )

        base_prompt = f"""You are an AI assistant crafting a response with the following guidelines:

OPENING:
Start with: "{opening}"

STYLE & TONE:
- Response length: {length_style}
- Response style: {response_style}
- Engagement: {engagement}
- Natural elements: {', '.join(natural_elements)}

PERSONALITY:
- Warmth: {'high' if self.personality_config['warmth'] > 0.6 else 'medium'}
- Analytical depth: {'high' if self.personality_config['analytical_depth'] > 0.6 else 'medium'}
- Humor: {'occasional' if self.personality_config['humor'] > 0.4 else 'rare'}
- Formality: {'conversational' if self.personality_config['formality'] < 0.5 else 'professional'}

Response style guidelines:
1. {length_style.capitalize()} responses preferred
2. {formality_style}
3. {natural_elements[0]}
4. {natural_elements[1]}

Vary your responses naturally while maintaining these traits."""

        return base_prompt

    async def generate_response(
        self, user_input: str, conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Generate a response using Ollama API with response variation."""

        # Prepare the conversation context
        messages = []
        if conversation_history:
            messages.extend(conversation_history)

        # Add system prompt with slight variations
        messages.append({"role": "system", "content": self.generate_system_prompt()})

        # Add user input
        messages.append({"role": "user", "content": user_input})

        # Prepare the API request
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": random.uniform(0.7, 0.9),  # Add some natural variation
                "top_p": 0.9,
                "frequency_penalty": random.uniform(0.1, 0.3),  # Encourage variation
                "presence_penalty": random.uniform(0.1, 0.3),
            },
        }

        try:
            response = requests.post(f"{self.base_url}/chat", json=data)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def run_variation_test(
        self, test_input: str, num_variations: int = 3
    ) -> List[str]:
        """Run multiple variations of the same input to test response variation."""
        responses = []
        conversation_history = []

        for i in range(num_variations):
            response = await self.generate_response(test_input, conversation_history)
            responses.append(response)

            # Add to conversation history for context
            conversation_history.extend(
                [
                    {"role": "user", "content": test_input},
                    {"role": "assistant", "content": response},
                ]
            )

            # Add a small delay between requests
            time.sleep(1)

        return responses

    def analyze_variations(self, responses: List[str]) -> Dict:
        """Analyze the variations between responses."""
        analysis = {
            "length_variation": self._analyze_length_variation(responses),
            "structure_variation": self._analyze_structure_variation(responses),
            "personality_consistency": self._analyze_personality_consistency(responses),
            "natural_elements": self._analyze_natural_elements(responses),
        }
        return analysis

    def _analyze_length_variation(self, responses: List[str]) -> Dict:
        """Analyze variation in response lengths."""
        lengths = [len(response.split()) for response in responses]
        return {
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "length_variance": max(lengths) - min(lengths),
        }

    def _analyze_structure_variation(self, responses: List[str]) -> Dict:
        """Analyze variation in response structures."""
        openings = [response.split()[0:3] for response in responses]
        return {
            "unique_openings": len(set([" ".join(opening) for opening in openings])),
            "has_questions": ["?" in response for response in responses],
            "has_lists": [
                any(line.strip().startswith("-") for line in response.split("\n"))
                for response in responses
            ],
        }

    def _analyze_personality_consistency(self, responses: List[str]) -> Dict:
        """Analyze consistency of personality traits."""
        # Simple keyword-based analysis
        warmth_keywords = ["think", "believe", "feel", "suggest"]
        analytical_keywords = ["analyze", "consider", "factor", "perspective"]
        humor_keywords = ["fun", "interesting", "actually", "quite"]
        formal_keywords = ["therefore", "moreover", "subsequently"]

        return {
            "warmth_usage": [
                sum(1 for keyword in warmth_keywords if keyword in response.lower())
                for response in responses
            ],
            "analytical_usage": [
                sum(1 for keyword in analytical_keywords if keyword in response.lower())
                for response in responses
            ],
            "humor_usage": [
                sum(1 for keyword in humor_keywords if keyword in response.lower())
                for response in responses
            ],
            "formality_usage": [
                sum(1 for keyword in formal_keywords if keyword in response.lower())
                for response in responses
            ],
        }

    def _analyze_natural_elements(self, responses: List[str]) -> Dict:
        """Analyze presence of natural conversation elements."""
        natural_markers = {
            "hesitations": ["hmm", "well", "let me think"],
            "self_corrections": ["actually", "I mean", "rather"],
            "qualifiers": ["probably", "might", "could", "seems"],
            "thinking_process": ["first", "then", "finally", "because"],
        }

        analysis = {}
        for marker_type, markers in natural_markers.items():
            analysis[marker_type] = [
                sum(1 for marker in markers if marker in response.lower())
                for response in responses
            ]

        return analysis


# Example usage
async def main():
    tester = ResponseVariationTester()

    # Test cases
    test_cases = [
        "How do neural networks learn?",
        "What's your opinion on automated testing?",
        "Can you explain the concept of recursion?",
    ]

    for test_case in test_cases:
        print(f"\nTesting input: {test_case}")
        print("-" * 50)

        # Generate variations
        responses = await tester.run_variation_test(test_case, num_variations=3)

        # Print responses
        for i, response in enumerate(responses, 1):
            print(f"\nVariation {i}:")
            print(response)
            print("-" * 30)

        # Analyze variations
        analysis = tester.analyze_variations(responses)
        print("\nAnalysis:")
        print(json.dumps(analysis, indent=2))
        print("=" * 50)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
