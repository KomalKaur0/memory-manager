"""
Core components of the AI Memory System.

This module contains the fundamental building blocks:
- MemoryNode: Individual memory units with connections
- MemoryGraph: Graph structure managing all memory nodes
- EdgeManager: Handles connection creation and management
- WeightCalculator: Computes and updates connection weights
"""

from .memory_node import MemoryNode, Connection, ConnectionType
from .memory_graph import MemoryGraph
from .edge_manager import EdgeManager
from .weight_calculator import WeightCalculator

__all__ = [
    # Core classes
    "MemoryNode",
    "MemoryGraph", 
    "EdgeManager",
    "WeightCalculator",
    
    # Data structures
    "Connection",
    "ConnectionType",
]

# Version information
__version__ = "0.1.0"
__author__ = "AI Memory System"

# Module-level constants
DEFAULT_DECAY_RATE = 0.01
DEFAULT_MAX_CONNECTIONS = 50
DEFAULT_CONNECTION_STRENGTH = 0.1
DEFAULT_MIN_WEIGHT = 0.05

# Connection type hierarchy for intelligent traversal
CONNECTION_HIERARCHIES = {
    ConnectionType.GENERAL_SPECIFIC: {
        "inverse": ConnectionType.SPECIFIC_GENERAL,
        "strength_factor": 1.0,
        "bidirectional": True
    },
    ConnectionType.SPECIFIC_GENERAL: {
        "inverse": ConnectionType.GENERAL_SPECIFIC,
        "strength_factor": 1.0,
        "bidirectional": True
    },
    ConnectionType.CAUSE_EFFECT: {
        "inverse": ConnectionType.EFFECT_CAUSE,
        "strength_factor": 0.9,
        "bidirectional": True
    },
    ConnectionType.EFFECT_CAUSE: {
        "inverse": ConnectionType.CAUSE_EFFECT,
        "strength_factor": 0.9,
        "bidirectional": True
    },
    ConnectionType.TEMPORAL_BEFORE: {
        "inverse": ConnectionType.TEMPORAL_AFTER,
        "strength_factor": 0.8,
        "bidirectional": False
    },
    ConnectionType.TEMPORAL_AFTER: {
        "inverse": ConnectionType.TEMPORAL_BEFORE,
        "strength_factor": 0.8,
        "bidirectional": False
    },
    ConnectionType.SIMILARITY: {
        "inverse": ConnectionType.SIMILARITY,
        "strength_factor": 1.0,
        "bidirectional": True
    },
    ConnectionType.CONTRAST: {
        "inverse": ConnectionType.CONTRAST,
        "strength_factor": 0.7,
        "bidirectional": True
    },
    ConnectionType.CONTEXT: {
        "inverse": ConnectionType.CONTEXT,
        "strength_factor": 0.6,
        "bidirectional": True
    }
}

def get_connection_strength_factor(connection_type: ConnectionType) -> float:
    """Get the strength factor for a connection type"""
    return CONNECTION_HIERARCHIES.get(connection_type, {}).get("strength_factor", 0.5)

def get_inverse_connection_type(connection_type: ConnectionType) -> ConnectionType:
    """Get the inverse connection type for bidirectional relationships"""
    return CONNECTION_HIERARCHIES.get(connection_type, {}).get("inverse", connection_type)

def is_bidirectional_connection(connection_type: ConnectionType) -> bool:
    """Check if a connection type is bidirectional"""
    return CONNECTION_HIERARCHIES.get(connection_type, {}).get("bidirectional", False)