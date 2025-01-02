"""
CoALA Memory System Implementation.

This module implements the memory architecture described in the CoALA paper,
adapted for virtual humans using LangChain components.
"""

from .working_memory import CoALAWorkingMemory
from .episodic_memory import CoALAEpisodicMemory
from .memory_system import CoALAMemorySystem
from .semantic_memory import CoALASemanticMemory
from .procedural_memory import CoALAProceduralMemory

__all__ = [
    "CoALAWorkingMemory",
    "CoALAEpisodicMemory",
    "CoALAMemorySystem",
    "CoALASemanticMemory",
    "CoALAProceduralMemory",
]
