import re
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from collections import defaultdict


@dataclass
class ResponseMetrics:
    """Metrics for analyzing response variability"""

    message_length: int
    sentence_count: int
    avg_sentence_length: float
    thinking_markers: List[str]
    correction_patterns: List[str]
    response_type: str  # 'direct', 'reflective', or 'mixed'
    messiness_score: float


class ResponseAnalyzer:
    def __init__(self):
        # Patterns voor verschillende aspecten van natuurlijke communicatie
        self.thinking_patterns = [
            r"\b(hmm|uhm|eh|oh)\b",
            r"\.{3,}",  # ... denk pauzes
            r"\(denkt\)",
            r"even nadenken",
        ]

        self.correction_patterns = [
            r"oh wacht",
            r"laat ik dat anders",
            r"ik bedoel",
            r"corrigeert",
            r"wat ik eigenlijk bedoel",
        ]

        self.reflective_markers = [
            r"ik denk",
            r"volgens mij",
            r"misschien",
            r"het lijkt erop",
            r"wat als we",
            r"interessant",
        ]

    def analyze_response(self, text: str) -> ResponseMetrics:
        """Analyseer een enkele response op verschillende metrics"""
        # Basis tekstanalyse
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        message_length = len(text)
        sentence_count = len(sentences)
        avg_sentence_length = (
            message_length / sentence_count if sentence_count > 0 else 0
        )

        # Zoek naar denkpatronen
        thinking_markers = []
        for pattern in self.thinking_patterns:
            matches = re.findall(pattern, text.lower())
            thinking_markers.extend(matches)

        correction_patterns = [
            pattern
            for pattern in self.correction_patterns
            if re.search(pattern, text.lower())
        ]
        # Bepaal response type
        reflective_count = sum(bool(re.search(pattern, text.lower()))
                           for pattern in self.reflective_markers)
        response_type = self._determine_response_type(reflective_count, len(sentences))

        # Bereken messiness score (0-1)
        messiness_score = self._calculate_messiness(
            len(thinking_markers), len(correction_patterns), sentence_count, text
        )

        return ResponseMetrics(
            message_length=message_length,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            thinking_markers=thinking_markers,
            correction_patterns=correction_patterns,
            response_type=response_type,
            messiness_score=messiness_score,
        )

    def analyze_conversation(self, messages: List[str]) -> Dict:
        """Analyseer een hele conversatie voor patronen"""
        metrics = [self.analyze_response(msg) for msg in messages]

        # Bereken conversation-level statistieken
        length_variation = np.std([m.message_length for m in metrics])
        response_types = [m.response_type for m in metrics]
        messiness_trend = [m.messiness_score for m in metrics]

        return {
            "individual_metrics": metrics,
            "length_variation": length_variation,
            "response_type_distribution": self._count_response_types(response_types),
            "avg_messiness": np.mean(messiness_trend),
            "messiness_trend": messiness_trend,
        }

    def _determine_response_type(
        self, reflective_count: int, sentence_count: int
    ) -> str:
        """Bepaal of een response direct, reflectief of gemengd is"""
        if sentence_count == 0:
            return "unknown"

        reflective_ratio = reflective_count / sentence_count
        if reflective_ratio > 0.6:
            return "reflective"
        elif reflective_ratio < 0.2:
            return "direct"
        else:
            return "mixed"

    def _calculate_messiness(
        self, thinking_count: int, correction_count: int, sentence_count: int, text: str
    ) -> float:
        """Bereken een genormaliseerde messiness score"""
        if sentence_count == 0:
            return 0.0

        # Basis componenten voor messiness
        thinking_ratio = min(thinking_count / sentence_count, 1.0)
        correction_ratio = min(correction_count / sentence_count, 1.0)

        # Check voor informele elementen
        informal_elements = len(
            re.findall(r"[!?]{2,}|\b(haha|nou|tja)\b", text.lower())
        )
        informal_ratio = min(informal_elements / sentence_count, 1.0)

        # Gewogen gemiddelde van verschillende factoren
        weights = [0.4, 0.3, 0.3]  # thinking, corrections, informal
        components = [thinking_ratio, correction_ratio, informal_ratio]

        return sum(w * c for w, c in zip(weights, components))

    def _count_response_types(self, types: List[str]) -> Dict[str, int]:
        """Tel de frequentie van verschillende response types"""
        counter = defaultdict(int)
        for t in types:
            counter[t] += 1
        return dict(counter)


# Voorbeeld gebruik
if __name__ == "__main__":
    analyzer = ResponseAnalyzer()

    # Test met een enkele response
    test_response = """
    Hmm... laat me daar even over nadenken. 
    Oh wacht, ik zie wat je bedoelt! 
    Volgens mij kunnen we dit het beste aanpakken door eerst...
    *corrigeert* Wat ik eigenlijk bedoel is dat we misschien moeten beginnen met een simpelere aanpak.
    """

    metrics = analyzer.analyze_response(test_response)
    print(f"Message length: {metrics.message_length}")
    print(f"Messiness score: {metrics.messiness_score:.2f}")
    print(f"Response type: {metrics.response_type}")
    print(f"Thinking markers found: {metrics.thinking_markers}")
    print(f"Correction patterns found: {metrics.correction_patterns}")
