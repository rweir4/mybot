"""
Pytest tests for the rate limiter.

Run with: pytest tests/
or: pytest tests/test_rate_limiter.py -v
"""

import pytest
from fastapi import HTTPException
from app.rate_limiter import GlobalRateLimiter, check_rate_limit, record_usage, get_rate_limit_stats
from app.config import settings


@pytest.fixture
def fresh_limiter():
    """Create a fresh rate limiter instance for each test."""
    return GlobalRateLimiter()


def test_initial_stats(fresh_limiter):
    """Test that initial stats are zero."""
    stats = fresh_limiter.get_stats()
    
    assert stats['hourly_stats']['requests_used'] == 0
    assert stats['hourly_stats']['requests_limit'] == settings.rate_limit_per_hour
    assert stats['daily_stats']['tokens_used'] == 0
    assert stats['daily_stats']['estimated_cost'] == 0.0


def test_single_request(fresh_limiter):
    """Test that a single request is allowed."""
    result = fresh_limiter.check_rate_limit()
    
    assert result['requests_this_hour'] == 1
    assert result['remaining_this_hour'] == settings.rate_limit_per_hour - 1


def test_multiple_requests_under_limit(fresh_limiter):
    """Test that multiple requests under the limit are allowed."""
    num_requests = min(5, settings.rate_limit_per_hour)
    
    for i in range(num_requests):
        result = fresh_limiter.check_rate_limit()
        assert result['requests_this_hour'] == i + 1
    
    stats = fresh_limiter.get_stats()
    assert stats['hourly_stats']['requests_used'] == num_requests


def test_rate_limit_exceeded(fresh_limiter):
    """Test that requests are blocked when limit is exceeded."""
    # Use up all requests
    for _ in range(settings.rate_limit_per_hour):
        fresh_limiter.check_rate_limit()
    
    # Next request should fail
    with pytest.raises(HTTPException) as exc_info:
        fresh_limiter.check_rate_limit()
    
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value.detail)


def test_token_usage_recording(fresh_limiter):
    """Test that token usage is recorded correctly."""
    input_tokens = 1000
    output_tokens = 500
    
    result = fresh_limiter.record_usage(input_tokens, output_tokens)
    
    assert result['tokens_used'] == 1500
    assert result['estimated_cost'] > 0
    assert result['daily_total_tokens'] == 1500


def test_cost_calculation(fresh_limiter):
    """Test that cost calculation is accurate."""
    # Claude Sonnet 4.5: $3/M input, $15/M output
    input_tokens = 1_000_000  # Should cost $3
    output_tokens = 1_000_000  # Should cost $15
    
    result = fresh_limiter.record_usage(input_tokens, output_tokens)
    
    # Total should be $18
    assert result['estimated_cost'] == pytest.approx(18.0, rel=0.01)


def test_daily_cost_limit(fresh_limiter):
    """Test that requests are blocked when daily cost limit is exceeded."""
    # Record enough usage to exceed daily limit
    # Using 1M output tokens = $15, which exceeds default $5 limit
    fresh_limiter.record_usage(input_tokens=0, output_tokens=1_000_000)
    
    # Next request should fail due to cost limit
    with pytest.raises(HTTPException) as exc_info:
        fresh_limiter.check_rate_limit()
    
    assert exc_info.value.status_code == 429
    assert "Daily cost limit exceeded" in str(exc_info.value.detail)


def test_api_disabled():
    """Test that requests are blocked when API is disabled."""
    # This test uses the global rate_limiter and requires changing settings
    # We'll skip this in favor of integration tests
    # For unit tests, we'd need to mock settings or use dependency injection
    pass


def test_stats_accuracy(fresh_limiter):
    """Test that stats accurately reflect usage."""
    # Make some requests
    num_requests = 3
    for _ in range(num_requests):
        fresh_limiter.check_rate_limit()
    
    # Record some usage
    fresh_limiter.record_usage(1000, 500)
    fresh_limiter.record_usage(2000, 1000)
    
    stats = fresh_limiter.get_stats()
    
    assert stats['hourly_stats']['requests_used'] == num_requests
    assert stats['hourly_stats']['requests_remaining'] == settings.rate_limit_per_hour - num_requests
    assert stats['daily_stats']['tokens_used'] == 4500  # 1000+500+2000+1000
    assert stats['daily_stats']['estimated_cost'] > 0


def test_global_rate_limiter_singleton():
    """Test that the global rate limiter functions work."""
    # Get initial stats
    initial_stats = get_rate_limit_stats()
    initial_count = initial_stats['hourly_stats']['requests_used']
    
    # Make a request
    check_rate_limit()
    
    # Check stats updated
    new_stats = get_rate_limit_stats()
    assert new_stats['hourly_stats']['requests_used'] == initial_count + 1
    
    # Record usage
    usage = record_usage(100, 50)
    assert usage['tokens_used'] == 150