import pytest
from fastapi import HTTPException
from app.rate_limiter import GlobalRateLimiter, rate_limiter
from app.config import settings


@pytest.fixture
def fresh_limiter():
    return GlobalRateLimiter()


def test_initial_stats(fresh_limiter):
    stats = fresh_limiter.get_stats()
    
    assert stats['hourly_stats']['requests_used'] == 0
    assert stats['hourly_stats']['requests_limit'] == settings.rate_limit_per_hour
    assert stats['daily_stats']['tokens_used'] == 0
    assert stats['daily_stats']['estimated_cost'] == 0.0


def test_single_request(fresh_limiter):
    result = fresh_limiter.check_rate_limit()
    
    assert result['requests_this_hour'] == 1
    assert result['remaining_this_hour'] == settings.rate_limit_per_hour - 1


def test_multiple_requests_under_limit(fresh_limiter):
    num_requests = min(5, settings.rate_limit_per_hour)
    
    for i in range(num_requests):
        result = fresh_limiter.check_rate_limit()
        assert result['requests_this_hour'] == i + 1
    
    stats = fresh_limiter.get_stats()
    assert stats['hourly_stats']['requests_used'] == num_requests


def test_rate_limit_exceeded(fresh_limiter):
    for _ in range(settings.rate_limit_per_hour):
        fresh_limiter.check_rate_limit()
    
    with pytest.raises(HTTPException) as exc_info:
        fresh_limiter.check_rate_limit()
    
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in str(exc_info.value.detail)


def test_token_usage_recording(fresh_limiter):
    input_tokens = 1000
    output_tokens = 500
    
    result = fresh_limiter.record_usage(input_tokens, output_tokens)
    
    assert result['tokens_used'] == 1500
    assert result['estimated_cost'] > 0
    assert result['daily_total_tokens'] == 1500


def test_cost_calculation(fresh_limiter):
    input_tokens = 1_000_000
    output_tokens = 1_000_000
    
    result = fresh_limiter.record_usage(input_tokens, output_tokens)
    
    assert result['estimated_cost'] == pytest.approx(18.0, rel=0.01)


def test_daily_cost_limit(fresh_limiter):
    fresh_limiter.record_usage(input_tokens=0, output_tokens=1_000_000)
    
    with pytest.raises(HTTPException) as exc_info:
        fresh_limiter.check_rate_limit()
    
    assert exc_info.value.status_code == 429
    assert "Daily cost limit exceeded" in str(exc_info.value.detail)


def test_stats_accuracy(fresh_limiter):
    num_requests = 3
    for _ in range(num_requests):
        fresh_limiter.check_rate_limit()
    
    fresh_limiter.record_usage(1000, 500)
    fresh_limiter.record_usage(2000, 1000)
    
    stats = fresh_limiter.get_stats()
    
    assert stats['hourly_stats']['requests_used'] == num_requests
    assert stats['hourly_stats']['requests_remaining'] == settings.rate_limit_per_hour - num_requests
    assert stats['daily_stats']['tokens_used'] == 4500
    assert stats['daily_stats']['estimated_cost'] > 0


def test_global_rate_limiter_singleton():
    initial_stats = rate_limiter.get_stats()
    initial_count = initial_stats['hourly_stats']['requests_used']
    
    rate_limiter.check_rate_limit()
    
    new_stats = rate_limiter.get_stats()
    assert new_stats['hourly_stats']['requests_used'] == initial_count + 1
    
    usage = rate_limiter.record_usage(100, 50)
    assert usage['tokens_used'] == 150