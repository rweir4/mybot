"""
Global rate limiter to protect against excessive API usage and control costs.

This implements a simple in-memory rate limiter that tracks:
- Total requests per hour (global across all users)
- Daily token usage and estimated costs
- Emergency kill switch check

For a single Railway instance, in-memory is sufficient. For multi-instance
deployments, consider Redis-based rate limiting.
"""

from datetime import datetime, timedelta
from typing import Dict, TypedDict
from fastapi import HTTPException, status
from app.config import settings
import threading


class HourlyStats(TypedDict):
    requests_used: int
    requests_limit: int
    requests_remaining: int
    resets_in_seconds: int
    resets_at: str


class DailyStats(TypedDict):
    tokens_used: int
    estimated_cost: float
    cost_limit: float
    cost_remaining: float
    resets_in_seconds: int
    resets_at: str


class RateLimitStats(TypedDict):
    api_enabled: bool
    hourly_stats: HourlyStats
    daily_stats: DailyStats


class UsageStats(TypedDict):
    requests_this_hour: int
    limit_per_hour: int
    remaining_this_hour: int
    daily_cost: float
    daily_cost_limit: float


class UsageRecord(TypedDict):
    tokens_used: int
    estimated_cost: float
    daily_total_tokens: int
    daily_total_cost: float


class RateLimitError(TypedDict):
    error: str
    message: str
    current_usage: str
    reset_in_minutes: int
    try_again_at: str


class CostLimitError(TypedDict):
    error: str
    message: str
    current_cost: str
    resets_at: str


class GlobalRateLimiter:
    """
    Global rate limiter with hourly request limits and daily cost tracking.
    Thread-safe for concurrent requests.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._reset_hourly_data()
        self._reset_daily_data()
    
    def _reset_hourly_data(self):
        self.hourly_requests = []
        self.hourly_reset_time = datetime.now() + timedelta(hours=1)
    
    def _reset_daily_data(self):
        self.daily_tokens_used = 0
        self.daily_estimated_cost = 0.0
        self.daily_reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
    
    def _clean_old_requests(self):
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.hourly_requests = [
            req_time for req_time in self.hourly_requests 
            if req_time > cutoff_time
        ]
    
    def _check_hourly_reset(self):
        if datetime.now() >= self.hourly_reset_time:
            self._reset_hourly_data()
    
    def _check_daily_reset(self):
        if datetime.now() >= self.daily_reset_time:
            self._reset_daily_data()
    
    def _check_api_enabled(self):
        if not settings.api_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=settings.emergency_message
            )
    
    def _check_hourly_limit(self, current_count: int):
        if current_count >= settings.rate_limit_per_hour:
            time_until_reset = (self.hourly_reset_time - datetime.now()).total_seconds()
            minutes_until_reset = int(time_until_reset / 60)
            
            error: RateLimitError = {
                "error": "Rate limit exceeded",
                "message": f"Global rate limit of {settings.rate_limit_per_hour} requests per hour reached",
                "current_usage": f"{current_count}/{settings.rate_limit_per_hour}",
                "reset_in_minutes": minutes_until_reset,
                "try_again_at": self.hourly_reset_time.isoformat()
            }
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error
            )
    
    def _check_cost_limit(self):
        if self.daily_estimated_cost >= settings.max_daily_cost:
            error: CostLimitError = {
                "error": "Daily cost limit exceeded",
                "message": f"Daily cost limit of ${settings.max_daily_cost} reached",
                "current_cost": f"${self.daily_estimated_cost:.2f}",
                "resets_at": self.daily_reset_time.isoformat()
            }
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error
            )
    
    def check_rate_limit(self) -> UsageStats:
        with self._lock:
            self._check_api_enabled()
            self._check_hourly_reset()
            self._check_daily_reset()
            self._clean_old_requests()
            
            current_count = len(self.hourly_requests)
            self._check_hourly_limit(current_count)
            self._check_cost_limit()
            
            self.hourly_requests.append(datetime.now())
            
            return {
                "requests_this_hour": current_count + 1,
                "limit_per_hour": settings.rate_limit_per_hour,
                "remaining_this_hour": settings.rate_limit_per_hour - current_count - 1,
                "daily_cost": self.daily_estimated_cost,
                "daily_cost_limit": settings.max_daily_cost
            }
    
    def record_usage(self, input_tokens: int, output_tokens: int) -> UsageRecord:
        with self._lock:
            self._check_daily_reset()
            
            # Claude Sonnet 4.5 pricing (as of Oct 2024)
            # Input: $3 per million tokens
            # Output: $15 per million tokens
            input_cost = (input_tokens / 1_000_000) * 3.0
            output_cost = (output_tokens / 1_000_000) * 15.0
            total_cost = input_cost + output_cost
            
            self.daily_tokens_used += (input_tokens + output_tokens)
            self.daily_estimated_cost += total_cost
            
            return {
                "tokens_used": input_tokens + output_tokens,
                "estimated_cost": total_cost,
                "daily_total_tokens": self.daily_tokens_used,
                "daily_total_cost": self.daily_estimated_cost
            }
    
    def get_stats(self) -> RateLimitStats:
        with self._lock:
            self._check_hourly_reset()
            self._check_daily_reset()
            self._clean_old_requests()
            
            current_count = len(self.hourly_requests)
            time_until_hourly_reset = (self.hourly_reset_time - datetime.now()).total_seconds()
            time_until_daily_reset = (self.daily_reset_time - datetime.now()).total_seconds()
            
            return {
                "api_enabled": settings.api_enabled,
                "hourly_stats": {
                    "requests_used": current_count,
                    "requests_limit": settings.rate_limit_per_hour,
                    "requests_remaining": settings.rate_limit_per_hour - current_count,
                    "resets_in_seconds": int(time_until_hourly_reset),
                    "resets_at": self.hourly_reset_time.isoformat()
                },
                "daily_stats": {
                    "tokens_used": self.daily_tokens_used,
                    "estimated_cost": round(self.daily_estimated_cost, 4),
                    "cost_limit": settings.max_daily_cost,
                    "cost_remaining": round(settings.max_daily_cost - self.daily_estimated_cost, 4),
                    "resets_in_seconds": int(time_until_daily_reset),
                    "resets_at": self.daily_reset_time.isoformat()
                }
            }


# Global singleton instance
rate_limiter = GlobalRateLimiter()


# Convenience functions for use in route handlers
def check_rate_limit() -> UsageStats:
    return rate_limiter.check_rate_limit()


def record_usage(input_tokens: int, output_tokens: int) -> UsageRecord:
    return rate_limiter.record_usage(input_tokens, output_tokens)


def get_rate_limit_stats() -> RateLimitStats:
    return rate_limiter.get_stats()