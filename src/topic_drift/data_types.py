from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ConversationData:
    """Container for raw conversation data."""

    conversations: List[Dict[str, any]]  # List of conversations with their turns
