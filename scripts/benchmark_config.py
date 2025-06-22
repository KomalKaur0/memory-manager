#!/usr/bin/env python3
"""
Benchmark Configuration for AI Memory System

This file contains configuration options for running different types of benchmarks
with various parameters to test system performance under different conditions.
"""

from typing import Dict, Any

# Default benchmark configuration
DEFAULT_CONFIG = {
    # Memory generation
    "num_memories": 1000,
    "num_operations": 500,
    "memory_content_length": 200,  # Average characters per memory content
    
    # Test data variety
    "num_topics": 15,
    "num_concepts": 20,
    "connection_probability": 0.3,  # Probability of creating connections between memories
    
    # Performance thresholds
    "min_ops_per_sec": 100,
    "min_success_rate": 0.95,
    "max_memory_usage_mb": 500,
    
    # Search configuration
    "search_terms": [
        "python", "machine", "data", "web", "database", 
        "cloud", "devops", "security", "ai", "ml"
    ],
    
    # Graph operations
    "max_connection_depth": 3,
    "similarity_threshold": 0.5,
    
    # Output configuration
    "save_results": True,
    "output_format": "json",
    "verbose_logging": True
}

# Small dataset configuration (for quick tests)
SMALL_CONFIG = {
    **DEFAULT_CONFIG,
    "num_memories": 100,
    "num_operations": 50,
    "memory_content_length": 100,
    "verbose_logging": False
}

# Large dataset configuration (for stress testing)
LARGE_CONFIG = {
    **DEFAULT_CONFIG,
    "num_memories": 10000,
    "num_operations": 2000,
    "memory_content_length": 500,
    "connection_probability": 0.5,
    "max_memory_usage_mb": 2000
}

# Memory usage stress test configuration
MEMORY_STRESS_CONFIG = {
    **DEFAULT_CONFIG,
    "num_memories": 5000,
    "num_operations": 1000,
    "memory_content_length": 1000,  # Large content to stress memory
    "connection_probability": 0.7,   # Many connections
    "max_memory_usage_mb": 5000
}

# Search performance configuration
SEARCH_PERFORMANCE_CONFIG = {
    **DEFAULT_CONFIG,
    "num_memories": 2000,
    "num_operations": 1000,
    "search_terms": [
        "python", "machine", "data", "web", "database", "cloud", "devops", 
        "security", "ai", "ml", "neural", "deep", "learning", "async",
        "microservices", "containerization", "kubernetes", "docker"
    ]
}

# Graph traversal configuration
TRAVERSAL_CONFIG = {
    **DEFAULT_CONFIG,
    "num_memories": 1500,
    "num_operations": 300,
    "connection_probability": 0.6,  # More connections for traversal testing
    "max_connection_depth": 5,
    "similarity_threshold": 0.3
}

# Available benchmark configurations
BENCHMARK_CONFIGS = {
    "default": DEFAULT_CONFIG,
    "small": SMALL_CONFIG,
    "large": LARGE_CONFIG,
    "memory_stress": MEMORY_STRESS_CONFIG,
    "search_performance": SEARCH_PERFORMANCE_CONFIG,
    "traversal": TRAVERSAL_CONFIG
}

def get_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Get benchmark configuration by name.
    
    Args:
        config_name: Name of the configuration to load
        
    Returns:
        Configuration dictionary
    """
    if config_name not in BENCHMARK_CONFIGS:
        print(f"Warning: Configuration '{config_name}' not found. Using default.")
        return DEFAULT_CONFIG
    
    return BENCHMARK_CONFIGS[config_name].copy()

def list_configs() -> None:
    """List all available benchmark configurations."""
    print("Available benchmark configurations:")
    print("=" * 50)
    
    for name, config in BENCHMARK_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  Memories: {config['num_memories']:,}")
        print(f"  Operations: {config['num_operations']:,}")
        print(f"  Content Length: {config['memory_content_length']} chars")
        print(f"  Connection Probability: {config['connection_probability']}")
        print(f"  Max Memory Usage: {config['max_memory_usage_mb']} MB")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate benchmark configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        "num_memories", "num_operations", "memory_content_length",
        "connection_probability", "min_ops_per_sec", "min_success_rate"
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key: {key}")
            return False
    
    # Validate numeric values
    if config["num_memories"] <= 0:
        print("Error: num_memories must be positive")
        return False
    
    if config["num_operations"] <= 0:
        print("Error: num_operations must be positive")
        return False
    
    if not 0 <= config["connection_probability"] <= 1:
        print("Error: connection_probability must be between 0 and 1")
        return False
    
    return True

if __name__ == "__main__":
    # Show available configurations when run directly
    list_configs() 