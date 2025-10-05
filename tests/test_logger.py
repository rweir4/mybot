import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from app.logger import UsageLogger, usage_logger


@pytest.fixture
def temp_log_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
        f.write("[]")
    
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def logger(temp_log_file):
    return UsageLogger(log_file_path=temp_log_file)


def test_logger_initialization(temp_log_file):
    logger = UsageLogger(log_file_path=temp_log_file)
    assert Path(temp_log_file).exists()
    
    # Check file contains empty array
    with open(temp_log_file, 'r') as f:
        data = json.load(f)
        assert data == []


def test_logger_creates_directory_if_not_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "nested" / "dir" / "logs.json"
        logger = UsageLogger(log_file_path=str(log_path))
        
        assert log_path.exists()
        assert log_path.parent.exists()


def test_log_single_request(logger, temp_log_file):
    logger.log_request(
        endpoint="/chat",
        input_tokens=1000,
        output_tokens=500,
        estimated_cost=0.0105,
        success=True
    )
    
    # Read log file
    with open(temp_log_file, 'r') as f:
        logs = json.load(f)
    
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/chat"
    assert logs[0]["tokens"]["input"] == 1000
    assert logs[0]["tokens"]["output"] == 500
    assert logs[0]["tokens"]["total"] == 1500
    assert logs[0]["estimated_cost_usd"] == 0.0105
    assert logs[0]["success"] is True
    assert "timestamp" in logs[0]


def test_log_multiple_requests(logger, temp_log_file):
    for i in range(5):
        logger.log_request(
            endpoint="/chat",
            input_tokens=100 * i,
            output_tokens=50 * i,
            estimated_cost=0.001 * i,
            success=True
        )
    
    with open(temp_log_file, 'r') as f:
        logs = json.load(f)
    
    assert len(logs) == 5


def test_log_failed_request(logger, temp_log_file):
    logger.log_request(
        endpoint="/chat",
        input_tokens=100,
        output_tokens=0,
        estimated_cost=0.0,
        success=False,
        error="Rate limit exceeded"
    )
    
    with open(temp_log_file, 'r') as f:
        logs = json.load(f)
    
    assert logs[0]["success"] is False
    assert logs[0]["error"] == "Rate limit exceeded"


def test_log_with_metadata(logger, temp_log_file):
    metadata = {
        "user_id": "test123",
        "model": "claude-sonnet-4-5",
        "retrieval_chunks": 3
    }
    
    logger.log_request(
        endpoint="/chat",
        input_tokens=1000,
        output_tokens=500,
        estimated_cost=0.01,
        success=True,
        metadata=metadata
    )
    
    with open(temp_log_file, 'r') as f:
        logs = json.load(f)
    
    assert logs[0]["metadata"] == metadata


def test_get_logs_all(logger):
    for i in range(3):
        logger.log_request(
            endpoint="/chat",
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.001,
            success=True
        )
    
    logs = logger.get_logs()
    assert len(logs) == 3


def test_get_logs_with_limit(logger):
    for i in range(10):
        logger.log_request(
            endpoint="/chat",
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.001,
            success=True
        )
    
    logs = logger.get_logs(limit=5)
    assert len(logs) == 5


def test_get_logs_returns_most_recent_first(logger):
    import time
    
    for i in range(3):
        logger.log_request(
            endpoint=f"/endpoint{i}",
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.001,
            success=True
        )
        time.sleep(0.01)  # Small delay
    
    logs = logger.get_logs()
    
    # Most recent should be first
    assert logs[0]["endpoint"] == "/endpoint2"
    assert logs[1]["endpoint"] == "/endpoint1"
    assert logs[2]["endpoint"] == "/endpoint0"


def test_get_stats_empty(logger):
    stats = logger.get_stats()
    
    assert stats["total_requests"] == 0
    assert stats["successful_requests"] == 0
    assert stats["failed_requests"] == 0
    assert stats["total_tokens"] == 0
    assert stats["total_cost_usd"] == 0.0


def test_get_stats_with_data(logger):
    logger.log_request("/chat", 1000, 500, 0.0105, success=True)
    logger.log_request("/chat", 2000, 1000, 0.021, success=True)
    logger.log_request("/chat", 100, 0, 0.0, success=False)
    
    stats = logger.get_stats()
    
    assert stats["total_requests"] == 3
    assert stats["successful_requests"] == 2
    assert stats["failed_requests"] == 1
    assert stats["total_tokens"] == 4600  # 1500 + 3000 + 100
    assert stats["total_cost_usd"] == pytest.approx(0.0315, rel=0.01)
    assert stats["average_tokens_per_request"] == pytest.approx(1533.33, rel=0.01)


def test_clear_logs(logger, temp_log_file):
    for i in range(5):
        logger.log_request(
            endpoint="/chat",
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.001,
            success=True
        )
    
    # Verify logs exist
    logs = logger.get_logs()
    assert len(logs) == 5
    
    # Clear logs
    logger.clear_logs()
    
    # Verify logs are cleared
    logs = logger.get_logs()
    assert len(logs) == 0
    
    # Verify file still exists and is valid JSON
    with open(temp_log_file, 'r') as f:
        data = json.load(f)
        assert data == []


def test_thread_safety(logger):
    import threading
    import time
    
    def log_many(thread_id):
        for i in range(5):
            logger.log_request(
                endpoint=f"/chat",
                input_tokens=100,
                output_tokens=50,
                estimated_cost=0.001,
                success=True,
                metadata={"thread_id": thread_id, "iteration": i}
            )
            time.sleep(0.005)  # Small delay to reduce extreme contention
    
    # Create multiple threads
    num_threads = 3
    logs_per_thread = 5
    expected_total = num_threads * logs_per_thread
    
    threads = [threading.Thread(target=log_many, args=(i,)) for i in range(num_threads)]
    
    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Give a moment for final writes to complete
    time.sleep(0.1)
    
    # Verify all logs were written
    logs = logger.get_logs()
    
    # Should have all 15 logs (3 threads × 5 logs each)
    assert len(logs) == expected_total, f"Expected {expected_total} logs, got {len(logs)}"
    
    # Verify no corruption - all logs should be valid
    for log in logs:
        assert "timestamp" in log
        assert "endpoint" in log
        assert log["tokens"]["total"] == 150
        assert "metadata" in log
        assert "thread_id" in log["metadata"]


def test_corrupted_file_recovery(temp_log_file):
    with open(temp_log_file, 'w') as f:
        f.write("{invalid json")
    
    logger = UsageLogger(log_file_path=temp_log_file)
    
    # Should still be able to log
    logger.log_request(
        endpoint="/chat",
        input_tokens=100,
        output_tokens=50,
        estimated_cost=0.001,
        success=True
    )
    
    # File should be fixed
    with open(temp_log_file, 'r') as f:
        data = json.load(f)
        assert len(data) == 1


def test_global_convenience_functions(temp_log_file):
    assert callable(usage_logger.log_request)
    assert callable(usage_logger.get_logs)
    assert callable(usage_logger.get_stats)
    assert callable(usage_logger.clear_logs)


def test_timestamp_format(logger, temp_log_file):
    logger.log_request(
        endpoint="/chat",
        input_tokens=100,
        output_tokens=50,
        estimated_cost=0.001,
        success=True
    )
    
    with open(temp_log_file, 'r') as f:
        logs = json.load(f)
    
    timestamp = logs[0]["timestamp"]
    
    # Should end with 'Z' (UTC)
    assert timestamp.endswith('Z')
    
    # Should be parseable as ISO format
    # Remove the 'Z' for parsing
    datetime.fromisoformat(timestamp[:-1])


def test_cost_rounding(logger, temp_log_file):
    logger.log_request(
        endpoint="/chat",
        input_tokens=100,
        output_tokens=50,
        estimated_cost=0.0123456789,
        success=True
    )
    
    with open(temp_log_file, 'r') as f:
        logs = json.load(f)
    
    # Should be rounded to 6 decimals
    assert logs[0]["estimated_cost_usd"] == 0.012346