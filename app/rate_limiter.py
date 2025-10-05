from datetime import datetime
from fastapi import HTTPException

class RateLimiter:
    def __init__(self):
        self.requests = []
    
    def check_rate_limit(self):
        # Check if under limit
        # Raise 429 if over
        pass