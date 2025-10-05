import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

import sys
sys.modules['langchain_openai'] = Mock()
sys.modules['langchain_anthropic'] = Mock()
sys.modules['langchain_pinecone'] = Mock()
sys.modules['pinecone'] = Mock()

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_rag_response():
    return {
        "answer": "Test answer from RAG",
        "sources": [
            {"content": "Source content 1", "source": "test1.pdf", "score": 0.95},
            {"content": "Source content 2", "source": "test2.pdf", "score": 0.85}
        ],
        "input_tokens": 100,
        "output_tokens": 50,
        "model": "claude-sonnet-4-5"
    }


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "api_enabled" in data
    assert "config_valid" in data


def test_health_shows_enabled_status(client):
    with patch('app.main.settings') as mock_settings:
        mock_settings.api_enabled = True
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_enabled"] is True


def test_health_shows_disabled_status(client):
    with patch('app.main.settings') as mock_settings:
        mock_settings.api_enabled = False
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "disabled"
        assert data["api_enabled"] is False


def test_chat_endpoint_success(client, mock_rag_response):
    with patch('app.main.query_rag') as mock_query, \
         patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request'):
        
        mock_query.return_value = mock_rag_response
        mock_limiter.check_rate_limit.return_value = {
            "requests_this_hour": 1,
            "limit_per_hour": 30,
            "remaining_this_hour": 29,
            "daily_cost": 0.0,
            "daily_cost_limit": 5.0
        }
        mock_limiter.record_usage.return_value = {
            "tokens_used": 150,
            "estimated_cost": 0.0015,
            "daily_total_tokens": 150,
            "daily_total_cost": 0.0015
        }
        
        response = client.post("/chat", json={"message": "What are your hobbies?"})
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "tokens_used" in data
        assert "estimated_cost" in data
        assert data["answer"] == "Test answer from RAG"


def test_chat_endpoint_validates_message(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_endpoint_requires_message(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_endpoint_validates_top_k(client):
    with patch('app.main.query_rag'), \
         patch('app.main.rate_limiter'), \
         patch('app.main.usage_logger.log_request'):
        
        response = client.post("/chat", json={"message": "test", "top_k": 0})
        assert response.status_code == 422
        
        response = client.post("/chat", json={"message": "test", "top_k": 25})
        assert response.status_code == 422


def test_chat_endpoint_respects_rate_limit(client):
    from fastapi import HTTPException
    
    with patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request') as mock_log:
        
        mock_limiter.check_rate_limit.side_effect = HTTPException(
            status_code=429,
            detail={"error": "Rate limit exceeded"}
        )
        
        response = client.post("/chat", json={"message": "test"})
        assert response.status_code == 429


def test_chat_endpoint_logs_success(client, mock_rag_response):
    with patch('app.main.query_rag') as mock_query, \
         patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request') as mock_log:
        
        mock_query.return_value = mock_rag_response
        mock_limiter.check_rate_limit.return_value = {}
        mock_limiter.record_usage.return_value = {"estimated_cost": 0.001}
        
        client.post("/chat", json={"message": "test"})
        
        mock_log.assert_called_once()
        call_args = mock_log.call_args.kwargs
        assert call_args["success"] is True
        assert call_args["endpoint"] == "/chat"


def test_chat_endpoint_logs_failure(client):
    from fastapi import HTTPException
    
    with patch('app.main.query_rag') as mock_query, \
         patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request') as mock_log:
        
        mock_limiter.check_rate_limit.return_value = {}
        mock_query.side_effect = Exception("Test error")
        
        response = client.post("/chat", json={"message": "test"})
        
        assert response.status_code == 500
        assert mock_log.call_count == 1
        call_args = mock_log.call_args.kwargs
        assert call_args["success"] is False


def test_chat_endpoint_truncates_long_sources(client, mock_rag_response):
    with patch('app.main.query_rag') as mock_query, \
         patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request'):
        
        mock_rag_response["sources"][0]["content"] = "x" * 500
        mock_query.return_value = mock_rag_response
        mock_limiter.check_rate_limit.return_value = {}
        mock_limiter.record_usage.return_value = {"estimated_cost": 0.001}
        
        response = client.post("/chat", json={"message": "test"})
        data = response.json()
        
        assert len(data["sources"][0]["content"]) <= 203


def test_chat_endpoint_includes_source_metadata(client, mock_rag_response):
    with patch('app.main.query_rag') as mock_query, \
         patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request'):
        
        mock_query.return_value = mock_rag_response
        mock_limiter.check_rate_limit.return_value = {}
        mock_limiter.record_usage.return_value = {"estimated_cost": 0.001}
        
        response = client.post("/chat", json={"message": "test"})
        data = response.json()
        
        source = data["sources"][0]
        assert "content" in source
        assert "source" in source
        assert "relevance_score" in source
        assert source["source"] == "test1.pdf"


def test_chat_endpoint_passes_top_k(client, mock_rag_response):
    with patch('app.main.query_rag') as mock_query, \
         patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request'):
        
        mock_query.return_value = mock_rag_response
        mock_limiter.check_rate_limit.return_value = {}
        mock_limiter.record_usage.return_value = {"estimated_cost": 0.001}
        
        client.post("/chat", json={"message": "test", "top_k": 7})
        
        mock_query.assert_called_once_with("test", 7)


def test_stats_endpoint(client):
    with patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger') as mock_logger:
        
        mock_limiter.get_stats.return_value = {
            "hourly_stats": {"requests_used": 10},
            "daily_stats": {"tokens_used": 5000}
        }
        mock_logger.get_stats.return_value = {
            "total_requests": 100,
            "total_cost_usd": 1.50
        }
        
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "rate_limit" in data
        assert "usage" in data
        assert data["rate_limit"]["hourly_stats"]["requests_used"] == 10
        assert data["usage"]["total_requests"] == 100

def test_chat_endpoint_returns_tokens_used(client, mock_rag_response):
    with patch('app.main.query_rag') as mock_query, \
         patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request'):
        
        mock_query.return_value = mock_rag_response
        mock_limiter.check_rate_limit.return_value = {}
        mock_limiter.record_usage.return_value = {"estimated_cost": 0.001}
        
        response = client.post("/chat", json={"message": "test"})
        data = response.json()
        
        assert data["tokens_used"] == 150


def test_chat_endpoint_returns_estimated_cost(client, mock_rag_response):
    with patch('app.main.query_rag') as mock_query, \
         patch('app.main.rate_limiter') as mock_limiter, \
         patch('app.main.usage_logger.log_request'):
        
        mock_query.return_value = mock_rag_response
        mock_limiter.check_rate_limit.return_value = {}
        mock_limiter.record_usage.return_value = {"estimated_cost": 0.0025}
        
        response = client.post("/chat", json={"message": "test"})
        data = response.json()
        
        assert data["estimated_cost"] == 0.0025