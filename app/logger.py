"""
Usage logging for tracking API requests, token usage, and costs.

Logs to two destinations:
1. Console (stdout) - visible in Railway logs
2. JSON file - persistent storage for historical analysis

The JSON log file can be read via the /stats endpoint.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from app.config import settings
import threading


class UsageLogger:
    """
    Thread-safe usage logger that writes to console and JSON file.
    """
    
    def __init__(self, log_file_path: Optional[str] = None):
        """
        Initialize the usage logger.
        
        Args:
            log_file_path: Path to JSON log file. Uses config default if None.
        """
        self.log_file_path = log_file_path or settings.log_file_path
        self._lock = threading.Lock()
        self._ensure_log_file_exists()
    
    def _ensure_log_file_exists(self):
        """Create log file and directory if they don't exist."""
        log_path = Path(self.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not log_path.exists():
            log_path.write_text("[]")  # Initialize with empty JSON array
    
    def log_request(
        self,
        endpoint: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a single API request.
        
        Args:
            endpoint: API endpoint that was called (e.g., "/chat")
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            estimated_cost: Estimated cost in USD
            success: Whether the request succeeded
            error: Error message if request failed
            metadata: Additional metadata to log
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "endpoint": endpoint,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "estimated_cost_usd": round(estimated_cost, 6),
            "success": success,
        }
        
        if error:
            log_entry["error"] = error
        
        if metadata:
            log_entry["metadata"] = metadata
        
        # Log to console (Railway logs)
        self._log_to_console(log_entry)
        
        # Log to JSON file
        self._log_to_file(log_entry)
    
    def _log_to_console(self, log_entry: Dict):
        """
        Log to console in a human-readable format.
        This appears in Railway logs.
        """
        timestamp = log_entry["timestamp"]
        endpoint = log_entry["endpoint"]
        tokens = log_entry["tokens"]["total"]
        cost = log_entry["estimated_cost_usd"]
        status = "✓" if log_entry["success"] else "✗"
        
        # Format: [2025-10-05T14:30:00Z] ✓ /chat | 1,234 tokens | $0.0123
        print(f"[{timestamp}] {status} {endpoint} | {tokens:,} tokens | ${cost:.4f}")
        
        if not log_entry["success"] and "error" in log_entry:
            print(f"  ERROR: {log_entry['error']}")
    
    def _log_to_file(self, log_entry: Dict):
        """
        Append log entry to JSON file.
        Thread-safe with file locking.
        """
        with self._lock:
            try:
                # Read existing logs
                with open(self.log_file_path, 'r') as f:
                    logs = json.load(f)
                
                # Append new entry
                logs.append(log_entry)
                
                # Write back
                with open(self.log_file_path, 'w') as f:
                    json.dump(logs, f, indent=2)
                    
            except json.JSONDecodeError:
                # File is corrupted, start fresh
                with open(self.log_file_path, 'w') as f:
                    json.dump([log_entry], f, indent=2)
            except Exception as e:
                # Log to console if file write fails
                print(f"⚠️  Failed to write to log file: {e}")
    
    def get_logs(
        self,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve logs from the JSON file.
        
        Args:
            limit: Maximum number of logs to return (most recent first)
            start_date: ISO format date to filter from (inclusive)
            end_date: ISO format date to filter to (inclusive)
        
        Returns:
            List of log entries
        """
        with self._lock:
            try:
                with open(self.log_file_path, 'r') as f:
                    logs = json.load(f)
                
                # Filter by date range if specified
                if start_date:
                    logs = [log for log in logs if log["timestamp"] >= start_date]
                if end_date:
                    logs = [log for log in logs if log["timestamp"] <= end_date]
                
                # Sort by timestamp (most recent first)
                logs.sort(key=lambda x: x["timestamp"], reverse=True)
                
                # Apply limit
                if limit:
                    logs = logs[:limit]
                
                return logs
                
            except (FileNotFoundError, json.JSONDecodeError):
                return []
    
    def get_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Get aggregated statistics from logs.
        
        Args:
            start_date: ISO format date to filter from
            end_date: ISO format date to filter to
        
        Returns:
            Dictionary with aggregated stats
        """
        logs = self.get_logs(start_date=start_date, end_date=end_date)
        
        if not logs:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "average_tokens_per_request": 0,
                "average_cost_per_request": 0.0
            }
        
        total_requests = len(logs)
        successful = sum(1 for log in logs if log["success"])
        failed = total_requests - successful
        total_tokens = sum(log["tokens"]["total"] for log in logs)
        total_cost = sum(log["estimated_cost_usd"] for log in logs)
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": failed,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "average_tokens_per_request": round(total_tokens / total_requests, 2),
            "average_cost_per_request": round(total_cost / total_requests, 6)
        }
    
    def clear_logs(self):
        """
        Clear all logs (use with caution!).
        Resets the log file to an empty array.
        """
        with self._lock:
            with open(self.log_file_path, 'w') as f:
                json.dump([], f)


# Global singleton instance
usage_logger = UsageLogger()


# Convenience functions for use in route handlers
def log_request(
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost: float,
    success: bool = True,
    error: Optional[str] = None,
    metadata: Optional[Dict] = None
):
    """Log an API request."""
    usage_logger.log_request(
        endpoint=endpoint,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
        success=success,
        error=error,
        metadata=metadata
    )


def get_logs(limit: Optional[int] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
    """Retrieve logs from storage."""
    return usage_logger.get_logs(limit=limit, start_date=start_date, end_date=end_date)


def get_usage_stats(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
    """Get aggregated usage statistics."""
    return usage_logger.get_stats(start_date=start_date, end_date=end_date)


def clear_logs():
    """Clear all logs."""
    usage_logger.clear_logs()