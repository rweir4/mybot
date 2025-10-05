"""
Configuration settings for the personal chatbot API.
Loads environment variables and provides centralized config access.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

REQUIRED_API_KEYS = [
    ('anthropic_api_key', 'ANTHROPIC_API_KEY'),
    ('pinecone_api_key', 'PINECONE_API_KEY'),
    ('openai_api_key', 'OPENAI_API_KEY'),
]

REQUIRED_POSITIVE_API_KEYS = [
    ('rate_limit_per_hour', 'RATE_LIMIT_PER_HOUR'),
    ('max_output_tokens', 'MAX_OUTPUT_TOKENS'),
]
    

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Create a .env file in the project root with these variables.
    See .env.example for a template.
    """
    
    # API Keys
    anthropic_api_key: str
    pinecone_api_key: str
    openai_api_key: str
    
    # Pinecone Configuration
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "personal-chatbot"
    
    # Rate Limiting
    rate_limit_per_hour: int = 30
    max_output_tokens: int = 4000
    max_daily_cost: float = 5.0
    
    # Emergency Kill Switch
    api_enabled: bool = True
    emergency_message: str = "API temporarily disabled for maintenance"
    
    # Claude Model Configuration
    claude_model: str = "claude-sonnet-4-5-20250929"
    claude_temperature: float = 0.7
    
    # Embedding Model Configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # RAG Configuration
    retrieval_top_k: int = 5  # Number of chunks to retrieve
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Logging
    log_file_path: str = "./usage_logs.json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create a singleton instance
settings = Settings()


def validate_config():
    errors = []
    
    for attr, name in REQUIRED_API_KEYS:
        value = getattr(settings, attr)
        if not value or value == "placeholder":
            errors.append(f"{name} is missing or placeholder")
    
    for attr, name in REQUIRED_POSITIVE_API_KEYS:
        if getattr(settings, attr) <= 0:
            errors.append(f"{name} must be positive")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return True


def get_config_summary():
    """
    Get a summary of current configuration (without exposing secrets).
    Useful for debugging and the /health endpoint.
    """
    return {
        "api_enabled": settings.api_enabled,
        "rate_limit_per_hour": settings.rate_limit_per_hour,
        "max_output_tokens": settings.max_output_tokens,
        "max_daily_cost": settings.max_daily_cost,
        "claude_model": settings.claude_model,
        "embedding_model": settings.embedding_model,
        "retrieval_top_k": settings.retrieval_top_k,
        "pinecone_index": settings.pinecone_index_name,
        "has_anthropic_key": bool(settings.anthropic_api_key and settings.anthropic_api_key != "placeholder"),
        "has_pinecone_key": bool(settings.pinecone_api_key and settings.pinecone_api_key != "placeholder"),
        "has_openai_key": bool(settings.openai_api_key and settings.openai_api_key != "placeholder"),
    }