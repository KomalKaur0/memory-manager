"""
Configuration Settings for AI Memory System
Handles environment variables and configuration management.
"""

from typing import Optional, Dict, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
from enum import Enum


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    Environment variables are prefixed with 'MEMORY_' by default.
    For example: MEMORY_NEO4J_URI, MEMORY_WEAVIATE_URL, etc.
    """
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_name: str = "AI Memory System"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Neo4j Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: Optional[str] = None
    neo4j_connection_timeout: int = 30
    neo4j_max_connection_pool_size: int = 50
    
    # Weaviate Settings
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None
    weaviate_timeout: int = 30
    
    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache_size: int = 1000
    embedding_batch_size: int = 32
    
    # Memory System Parameters
    decay_rate: float = 0.01
    connection_strength_increment: float = 0.1
    min_connection_weight: float = 0.05
    max_connection_weight: float = 1.0
    
    # Search Parameters
    max_search_depth: int = 3
    default_search_limit: int = 10
    semantic_search_threshold: float = 0.7
    hybrid_search_alpha: float = 0.75
    
    # Import/Export Settings
    import_similarity_threshold: float = 0.85
    auto_connect_threshold: float = 0.7
    export_batch_size: int = 1000
    
    # File Storage
    data_dir: Path = Path("data")
    migrations_dir: Path = Path("migrations")
    logs_dir: Path = Path("logs")
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_cors_origins: list[str] = ["*"]
    api_rate_limit: int = 100
    
    # Performance Tuning
    cache_enabled: bool = True
    cache_ttl: int = 3600
    batch_processing_size: int = 100
    max_concurrent_operations: int = 10
    
    # Logging Settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_rotation: str = "midnight"
    log_retention_days: int = 30
    
    # Feature Flags
    enable_auto_connections: bool = True
    enable_temporal_decay: bool = True
    enable_duplicate_detection: bool = True
    enable_analytics: bool = True
    
    @field_validator("data_dir", "migrations_dir", "logs_dir", mode="after")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("neo4j_uri")
    @classmethod
    def validate_neo4j_uri(cls, v: str) -> str:
        """Validate Neo4j URI format."""
        if not v.startswith(("bolt://", "neo4j://", "bolt+s://", "neo4j+s://")):
            raise ValueError(f"Invalid Neo4j URI: {v}")
        return v
    
    @field_validator("weaviate_url")
    @classmethod
    def validate_weaviate_url(cls, v: str) -> str:
        """Validate Weaviate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid Weaviate URL: {v}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()
    
    @field_validator("decay_rate", "connection_strength_increment", "min_connection_weight", "max_connection_weight", "semantic_search_threshold", "hybrid_search_alpha", "import_similarity_threshold", "auto_connect_threshold")
    @classmethod
    def validate_float_range(cls, v: float) -> float:
        """Validate float values are in range [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError(f"Value must be between 0 and 1, got {v}")
        return v
    
    @field_validator("max_search_depth")
    @classmethod
    def validate_search_depth(cls, v: int) -> int:
        """Validate search depth is reasonable."""
        if not 1 <= v <= 10:
            raise ValueError(f"Search depth must be between 1 and 10, got {v}")
        return v
    
    @field_validator("api_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """Get Neo4j configuration as dict."""
        return {
            "uri": self.neo4j_uri,
            "auth": (self.neo4j_username, self.neo4j_password),
            "database": self.neo4j_database,
            "max_connection_lifetime": 3600,
            "max_connection_pool_size": self.neo4j_max_connection_pool_size,
            "connection_acquisition_timeout": self.neo4j_connection_timeout,
        }
    
    def get_weaviate_config(self) -> Dict[str, Any]:
        """Get Weaviate configuration as dict."""
        config = {
            "url": self.weaviate_url,
            "timeout_config": (5, self.weaviate_timeout),
        }
        if self.weaviate_api_key:
            config["auth_client_secret"] = self.weaviate_api_key
        return config
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    def get_connection_params(self) -> Dict[str, float]:
        """Get memory system connection parameters."""
        return {
            "decay_rate": self.decay_rate,
            "connection_strength_increment": self.connection_strength_increment,
            "min_connection_weight": self.min_connection_weight,
            "max_connection_weight": self.max_connection_weight,
        }
    
    def get_search_params(self) -> Dict[str, Any]:
        """Get search-related parameters."""
        return {
            "max_depth": self.max_search_depth,
            "default_limit": self.default_search_limit,
            "semantic_threshold": self.semantic_search_threshold,
            "hybrid_alpha": self.hybrid_search_alpha,
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance (singleton).
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None


def load_settings_from_file(file_path: str) -> Settings:
    """
    Load settings from a specific .env file.
    
    Args:
        file_path: Path to .env file
        
    Returns:
        Settings instance
    """
    return Settings()


# Environment-specific configuration presets
DEVELOPMENT_OVERRIDES = {
    "debug": True,
    "log_level": "DEBUG",
    "api_cors_origins": ["*"],
    "cache_ttl": 60,  # Shorter cache in dev
}

PRODUCTION_OVERRIDES = {
    "debug": False,
    "log_level": "INFO",
    "api_cors_origins": ["https://yourdomain.com"],
    "api_workers": 8,
    "cache_ttl": 3600,
    "log_to_file": True,
}

TESTING_OVERRIDES = {
    "neo4j_uri": "bolt://localhost:7688",  # Different port for test DB
    "weaviate_url": "http://localhost:8081",
    "log_level": "DEBUG",
    "log_to_file": False,
    "enable_analytics": False,
}


def get_environment_settings(env: Environment) -> Settings:
    """
    Get settings with environment-specific overrides.
    
    Args:
        env: Environment to use (defaults to current)
        
    Returns:
        Settings instance with overrides applied
    """
    base_settings = get_settings()
    env = env or base_settings.environment
    
    overrides = {}
    if env == Environment.DEVELOPMENT:
        overrides = DEVELOPMENT_OVERRIDES
    elif env == Environment.PRODUCTION:
        overrides = PRODUCTION_OVERRIDES
    elif env == Environment.TESTING:
        overrides = TESTING_OVERRIDES
    
    # Apply overrides
    settings_dict = base_settings.model_dump()
    settings_dict.update(overrides)
    
    return Settings(**settings_dict)