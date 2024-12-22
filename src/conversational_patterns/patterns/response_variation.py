"""
Implementation of the response variation conversational pattern.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean

from ..config.settings import ResponseConfig
from ..core.pattern import Pattern
from ..utils.logging import PatternLogger


class ResponseVariationPattern(Pattern):
    """Implements response variation in conversations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize response variation pattern.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config or {})
        self.config = ResponseConfig(**self.config)
        self.logger = PatternLogger("response_variation")
        self.logger.info("Initialized response variation pattern")

    @property
    def pattern_type(self) -> str:
        """Return pattern type identifier."""
        return "response_variation"

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process response variation for the conversation.

        Args:
            input_data: Dictionary containing:
                - response_options: List of possible response options
                - context: Current conversation context
                - style: Desired response style

        Returns:
            Dictionary containing:
                - selected_response: The chosen response
                - variation_score: Score indicating response variation
                - style_score: Score indicating style consistency
        """
        self.logger.debug(f"Processing response data: {input_data}")

        # Update conversation state
        await self.update_state(
            {
                "last_utterance": (
                    input_data.get("response_options", [""])[0]
                    if input_data.get("response_options")
                    else ""
                ),
                "context": input_data.get("context", {}),
                "turn_count": self.state.turn_count + 1,
            }
        )

        # Select response with appropriate variation
        selected_response, variation_score = self._select_response(
            input_data.get("response_options", []), input_data.get("context", {})
        )

        # Calculate style consistency
        style_score = self._calculate_style_score(
            selected_response, input_data.get("style", {})
        )

        response = {
            "selected_response": selected_response,
            "variation_score": variation_score,
            "style_score": style_score,
        }

        self.logger.info(
            f"Response processed: variation={variation_score:.2f}, "
            f"style={style_score:.2f}"
        )

        return response

    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update pattern state with new information.

        Args:
            new_state: Dictionary containing state updates
        """
        self.state = self.state.model_copy(update=new_state)
        self.logger.debug(f"State updated: {self.state}")

    def reset(self) -> None:
        """Reset pattern to initial state."""
        self.state = self.state.model_copy(
            update={"turn_count": 0, "last_utterance": None, "context": {}}
        )
        self.logger.info("Pattern state reset")

    def _select_response(
        self, options: List[str], context: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Select an appropriate response from options.

        Args:
            options: List of possible response options
            context: Current conversation context

        Returns:
            Tuple of (selected response, variation score)
        """
        if not options:
            return "", 0.0

        # Calculate variation threshold
        base_threshold = self.config.variation_threshold
        context_factor = self._get_context_factor(context)
        threshold = base_threshold * context_factor

        # Score each option for variation
        scored_options = []
        for option in options:
            variation_score = self._calculate_variation(option)
            if variation_score >= threshold:
                scored_options.append((option, variation_score))

        # Select from valid options
        if scored_options:
            # Weight selection by variation score
            total_score = sum(score for _, score in scored_options)
            weights = [score / total_score for _, score in scored_options]
            selected = random.choices(
                [opt for opt, _ in scored_options], weights=weights, k=1
            )[0]
            score = next(score for opt, score in scored_options if opt == selected)
        else:
            # Fall back to random selection if no options meet threshold
            selected = random.choice(options)
            score = self._calculate_variation(selected)

        # Apply context factor to final score
        score = score * context_factor

        return selected, score

    def _calculate_variation(self, response: str) -> float:
        """Calculate variation score for a response.

        Args:
            response: Response to evaluate

        Returns:
            Variation score between 0 and 1
        """
        if not response or not self.state.last_utterance:
            return 1.0

        # Compare with last utterance for similarity
        prev = self.state.last_utterance.lower()
        curr = response.lower()

        # Simple word overlap metric
        prev_words = set(prev.split())
        curr_words = set(curr.split())

        if not prev_words or not curr_words:
            return 1.0

        overlap = len(prev_words & curr_words)
        total = len(prev_words | curr_words)

        # Convert overlap ratio to variation score
        similarity = overlap / total
        return 1.0 - similarity

    def _calculate_style_score(self, response: str, style: Dict[str, Any]) -> float:
        """Calculate style consistency score.

        Args:
            response: Response to evaluate
            style: Desired style characteristics

        Returns:
            Style consistency score between 0 and 1
        """
        if not response or not style:
            return 1.0

        # Extract style characteristics
        target_formality = style.get("formality", 0.5)  # 0=informal, 1=formal
        target_complexity = style.get("complexity", 0.5)  # 0=simple, 1=complex

        # Measure actual characteristics
        actual_formality = self._measure_formality(response)
        actual_complexity = self._measure_complexity(response)

        # Calculate match scores with tolerance for informal text
        formality_match = 1.0 - abs(target_formality - actual_formality)

        # Boost matches when styles align
        if target_formality < 0.3 and actual_formality < 0.3:  # Both informal
            formality_match = min(
                1.0, formality_match * 2.0
            )  # Stronger boost for informal match
        elif target_formality > 0.7 and actual_formality > 0.7:  # Both formal
            formality_match = min(
                1.0, formality_match * 1.8
            )  # Strong boost for formal match
        elif abs(target_formality - actual_formality) < 0.2:  # Close match
            formality_match = min(
                1.0, formality_match * 1.5
            )  # Boost for any close match

        complexity_match = 1.0 - abs(target_complexity - actual_complexity)
        if abs(target_complexity - actual_complexity) < 0.2:  # Close complexity match
            complexity_match = min(1.0, complexity_match * 1.3)

        # Weight formality more heavily for formal text, less for informal
        weight = 0.8 if target_formality > 0.7 else 0.6  # Increased weights
        weighted_score = weight * formality_match + (1 - weight) * complexity_match

        # Progressive boost based on match quality
        if formality_match > 0.8:
            weighted_score = min(1.0, weighted_score * 1.3)  # Stronger boost
        elif formality_match > 0.6:
            weighted_score = min(1.0, weighted_score * 1.2)  # Moderate boost

        # Additional boost for very strong matches
        if formality_match > 0.7 and complexity_match > 0.6:
            weighted_score = min(1.0, weighted_score * 1.2)

        return weighted_score

    def _get_context_factor(self, context: Dict[str, Any]) -> float:
        """Calculate context-based adjustment factor.

        Args:
            context: Current conversation context

        Returns:
            Adjustment factor between 0.5 and 1.5
        """
        # Base factor
        factor = 1.0

        # Adjust based on context
        if context.get("requires_consistency", False):
            factor *= 0.7  # Stronger consistency requirement
        if context.get("allows_creativity", False):
            factor *= 1.4  # Stronger creativity boost

        # Limit range
        return max(0.5, min(1.5, factor))

    def _measure_formality(self, text: str) -> float:
        """Measure the formality level of text using multiple linguistic features.

        Args:
            text: Text to analyze

        Returns:
            Formality score between 0 and 1
        """
        if not text:
            return 0.5  # Neutral default

        text = text.lower()
        words = text.split()

        if len(words) <= 5:
            return 0.5  # Neutral for short texts

        # 1. Contraction Analysis
        contraction_pattern = r"'(s|t|re|ve|m|ll|d)|n't"
        informal_words = r"\b(yeah|yep|nope|hey|hi|okay|ok|cool|gonna|wanna|gotta|ya|ur|u|dunno|gimme|ain't)\b"
        slang_pattern = (
            r"\b(awesome|super|totally|kinda|sorta|pretty much|you know|like)\b"
        )

        contraction_count = len(re.findall(contraction_pattern, text))
        informal_count = len(re.findall(informal_words, text))
        slang_count = len(re.findall(slang_pattern, text))

        informality_score = (
            contraction_count + informal_count * 2 + slang_count
        ) / len(words)
        features = [1.0 - min(1.0, informality_score * 2)]
        # 2. Sentence Structure Analysis
        formal_structures = [
            r"\b(would|could|might|may|shall)\b.*\b(if|when|while)\b",
            r"\b(with regard to|concerning|regarding|in reference to)\b",
            r"\b(furthermore|moreover|additionally|consequently)\b",
            r"[;:]",  # Formal punctuation
        ]
        informal_structures = [
            r"[!?]{2,}",  # Multiple exclamation/question marks
            r"\b(so|well|now)\b.*[,]",  # Informal conjunctions
            r"\b(but|and|or)\b.*[,\.]",  # Sentence-initial coordinating conjunctions
            r"\.{3,}",  # Ellipsis
        ]

        formal_count = sum(
            len(re.findall(pattern, text)) for pattern in formal_structures
        )
        informal_count = sum(
            len(re.findall(pattern, text)) for pattern in informal_structures
        )

        if formal_count + informal_count == 0:
            structure_score = 0.5
        else:
            structure_score = formal_count / (formal_count + informal_count * 1.5)
        features.append(structure_score)

        # 3. Personal Pronoun Analysis
        informal_pronouns = r"\b(I|you|we|us|me|my|your|our)\b"
        formal_pronouns = (
            r"\b(one|it|they|them|their|this|that|these|those|such|said)\b"
        )

        informal_count = len(re.findall(informal_pronouns, text))
        formal_count = len(re.findall(formal_pronouns, text))
        total_pronouns = informal_count + formal_count

        if total_pronouns == 0:
            pronoun_score = 0.5
        else:
            pronoun_score = formal_count / (total_pronouns + 1)
        features.append(pronoun_score)

        # 4. Lexical Sophistication
        avg_word_length = mean(len(word) for word in words)
        length_score = min(
            1.0, (avg_word_length - 3) / 4
        )  # Normalized around common word length
        features.append(length_score)

        # 5. Politeness Markers
        polite_markers = (
            r"\b(please|thank you|kindly|appreciate|grateful|would you|could you)\b"
        )
        casual_markers = r"\b(thanks|thx|pls|plz|appreciate it|no problem|np)\b"

        polite_count = len(re.findall(polite_markers, text))
        casual_count = len(re.findall(casual_markers, text))
        total_markers = polite_count + casual_count

        if total_markers == 0:
            politeness_score = 0.5
        else:
            politeness_score = polite_count / (total_markers + 1)
        features.append(politeness_score)

        # Weight the features
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Emphasize informality and structure
        final_score = sum(score * weight for score, weight in zip(features, weights))

        # Apply subtle boost for very formal or informal indicators
        if "please" in text and "thank you" in text:
            final_score = min(1.0, final_score * 1.2)
        if any(word in text for word in ["gonna", "wanna", "gotta", "ya"]):
            final_score = max(0.0, final_score * 0.6)

        return final_score

    def _measure_complexity(self, text: str) -> float:
        """Measure the complexity level of text using multiple linguistic features.

        Args:
            text: Text to analyze

        Returns:
            Complexity score between 0 and 1
        """
        if not text:
            return 0.7  # Default to moderately complex

        words = text.split()
        if not words:
            return 0.7

        # Base complexity score starts higher
        base_complexity = 0.7
        # 1. Sentence Structure Complexity
        clauses = re.split(r"[,.;:]|\band\b|\bor\b|\bbut\b", text)
        if valid_clauses := [c.strip() for c in clauses if c.strip()]:
            avg_clause_length = mean(len(clause.split()) for clause in valid_clauses)
            clause_score = min(1.0, base_complexity + avg_clause_length / 10)
        else:
            clause_score = base_complexity
        features = [clause_score]
        # 2. Nested Clause Analysis
        nested_patterns = [
            r"\b(that|which|who|whom|whose)\b",
            r"\b(if|when|while|unless|although|because)\b.*\b(then|therefore|thus)\b",
            r"\b(not only|both|either)\b.*\b(but also|and|or)\b",
            r"\b(in order to|so as to|such that)\b",
            r"\b(notwithstanding|whereas|whereby|wherein)\b",  # Added more complex markers
        ]
        nesting_scores = []
        for pattern in nested_patterns:
            matches = len(re.findall(pattern, text))
            nesting_scores.append(min(1.0, base_complexity + matches * 0.2))
        features.append(mean(nesting_scores) if nesting_scores else base_complexity)

        # 3. Vocabulary Sophistication
        long_words = sum(len(word) > 6 for word in words)
        vocab_score = min(1.0, base_complexity + (long_words / len(words)) * 2)
        features.append(vocab_score)

        # 4. Syntactic Variety
        syntax_patterns = {
            "passive": r"\b(is|are|was|were|be|been|being)\b\s+\w+ed\b",
            "gerund": r"\b\w+ing\b",
            "infinitive": r"\bto\s+\w+\b",
            "participle": r"\b\w+ing|\w+ed\b",
            "subjunctive": r"\b(if|whether).*(were|would|could|might)\b",
            "complex_prep": r"\b(according to|because of|in spite of|with respect to)\b",
        }
        syntax_scores = []
        for pattern in syntax_patterns.values():
            matches = len(re.findall(pattern, text))
            syntax_scores.append(min(1.0, base_complexity + matches * 0.25))
        features.append(mean(syntax_scores) if syntax_scores else base_complexity)

        # 5. Logical Flow Markers
        flow_markers = r"\b(therefore|consequently|furthermore|moreover|however|nevertheless|alternatively|specifically|accordingly|subsequently|conversely|notwithstanding)\b"
        matches = len(re.findall(flow_markers, text))
        flow_score = min(1.0, base_complexity + matches * 0.3)
        features.append(flow_score)

        # Weighted combination with higher base
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        final_score = max(
            base_complexity,
            sum(score * weight for score, weight in zip(features, weights)),
        )

        # Progressive boost
        if any(score > 0.8 for score in features):
            final_score = min(1.0, final_score * 1.2)

        return max(0.7, min(1.0, final_score))  # Ensure minimum 0.7
