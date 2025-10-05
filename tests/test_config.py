"""
Tests for configuration loading and validation.

Run with: pytest tests/test_config.py -v
"""

import pytest
import os
from unittest.mock import patch
from app.config import Settings, validate_config, get_config_summary


def test_config_loads_from_env():
    """Test that config successfully loads from .env file."""
    from app.config import settings
    
    # These should be loaded from your .env file
    assert settings.anthropic_api_key is not None
    assert settings.pinecone_api_key is not None
    assert settings.openai_api_key is not None


def test_config_has_defaults():
    """Test that config has sensible defaults."""
    from app.config import settings
    
    assert settings.rate_limit_per_hour > 0
    assert settings.max_output_tokens > 0
    assert settings.max_daily_cost > 0
    assert settings.claude_model == "claude-sonnet-4-5-20250929"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.retrieval_top_k > 0
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0


def test_config_types():
    """Test that config values have correct types."""
    from app.config import settings
    
    assert isinstance(settings.anthropic_api_key, str)
    assert isinstance(settings.rate_limit_per_hour, int)
    assert isinstance(settings.max_output_tokens, int)
    assert isinstance(settings.max_daily_cost, float)
    assert isinstance(settings.api_enabled, bool)
    assert isinstance(settings.claude_temperature, float)


def test_validate_config_success():
    """Test that validate_config passes with valid configuration."""
    # Should not raise any exceptions with proper .env file
    try:
        validate_config()
        assert True
    except ValueError:
        pytest.fail("validate_config() raised ValueError with valid config")


def test_validate_config_fails_without_api_keys():
    """Test that validate_config fails when API keys are missing."""
    # Create a Settings instance with placeholder keys
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'placeholder',
        'PINECONE_API_KEY': 'real-key',
        'OPENAI_API_KEY': 'real-key'
    }):
        temp_settings = Settings()
        
        # Mock the settings module to use our temp settings
        with patch('app.config.settings', temp_settings):
            with pytest.raises(ValueError) as exc_info:
                validate_config()
            
            assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_validate_config_fails_with_invalid_limits():
    """Test that validate_config fails with invalid rate limits."""
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'real-key',
        'PINECONE_API_KEY': 'real-key',
        'OPENAI_API_KEY': 'real-key',
        'RATE_LIMIT_PER_HOUR': '0'  # Invalid
    }):
        temp_settings = Settings()
        
        with patch('app.config.settings', temp_settings):
            with pytest.raises(ValueError) as exc_info:
                validate_config()
            
            assert "RATE_LIMIT_PER_HOUR" in str(exc_info.value)


def test_get_config_summary():
    """Test that config summary returns expected structure."""
    summary = get_config_summary()
    
    # Check required keys exist
    assert 'api_enabled' in summary
    assert 'rate_limit_per_hour' in summary
    assert 'max_output_tokens' in summary
    assert 'claude_model' in summary
    assert 'has_anthropic_key' in summary
    assert 'has_pinecone_key' in summary
    assert 'has_openai_key' in summary
    
    # Check types
    assert isinstance(summary['api_enabled'], bool)
    assert isinstance(summary['rate_limit_per_hour'], int)
    assert isinstance(summary['has_anthropic_key'], bool)


def test_config_summary_hides_secrets():
    """Test that config summary doesn't expose API keys."""
    summary = get_config_summary()
    
    # Should NOT contain actual API keys
    assert 'anthropic_api_key' not in summary
    assert 'pinecone_api_key' not in summary
    assert 'openai_api_key' not in summary
    
    # Should only contain boolean flags
    assert summary['has_anthropic_key'] in [True, False]
    assert summary['has_pinecone_key'] in [True, False]
    assert summary['has_openai_key'] in [True, False]


def test_environment_override():
    """Test that environment variables can override defaults."""
    # Set a custom value
    custom_limit = 99
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-key',
        'PINECONE_API_KEY': 'test-key',
        'OPENAI_API_KEY': 'test-key',
        'RATE_LIMIT_PER_HOUR': str(custom_limit)
    }):
        temp_settings = Settings()
        assert temp_settings.rate_limit_per_hour == custom_limit


def test_kill_switch_default():
    """Test that API is enabled by default."""
    from app.config import settings
    assert settings.api_enabled is True


def test_temperature_in_valid_range():
    """Test that Claude temperature is in valid range (0.0 to 1.0)."""
    from app.config import settings
    assert 0.0 <= settings.claude_temperature <= 1.0


def test_chunk_overlap_less_than_size():
    """Test that chunk overlap is less than chunk size."""
    from app.config import settings
    assert settings.chunk_overlap < settings.chunk_size


@pytest.mark.parametrize("env_var,expected_type", [
    ("RATE_LIMIT_PER_HOUR", int),
    ("MAX_OUTPUT_TOKENS", int),
    ("MAX_DAILY_COST", float),
    ("API_ENABLED", bool),
    ("CLAUDE_TEMPERATURE", float),
])
def test_config_type_coercion(env_var, expected_type):
    """Test that config correctly coerces types from environment variables."""
    test_values = {
        int: "42",
        float: "3.14",
        bool: "true"
    }
    
    env_dict = {
        'ANTHROPIC_API_KEY': 'test',
        'PINECONE_API_KEY': 'test',
        'OPENAI_API_KEY': 'test',
        env_var: test_values[expected_type]
    }
    
    with patch.dict(os.environ, env_dict):
        temp_settings = Settings()
        value = getattr(temp_settings, env_var.lower())
        assert isinstance(value, expected_type)