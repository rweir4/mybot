import json
from datetime import datetime

def log_usage(tokens: int, cost: float):
    # Print to console (Railway logs)
    print(f"[USAGE] Tokens: {tokens}, Cost: ${cost:.4f}")
    
    # Append to JSON file
    # (stored on Railway volume)