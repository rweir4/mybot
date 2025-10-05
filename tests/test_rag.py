import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
sys.modules['langchain_openai'] = Mock()
sys.modules['langchain_anthropic'] = Mock()
sys.modules['langchain_pinecone'] = Mock()
sys.modules['pinecone'] = Mock()

from app.rag import RAGEngine, query_rag, get_rag_engine, RetrievedChunk, RAGResponse, SYSTEM_PROMPT


@pytest.fixture
def mock_vectorstore():
    mock = Mock()
    mock.similarity_search_with_score = Mock(return_value=[])
    return mock


@pytest.fixture
def mock_anthropic_client():
    mock = Mock()
    response = Mock()
    response.content = [Mock(text="This is a test answer")]
    response.usage = Mock(input_tokens=100, output_tokens=50)
    mock.messages.create = Mock(return_value=response)
    return mock


@pytest.fixture
def rag_engine(mock_vectorstore, mock_anthropic_client):
    with patch('app.rag.OpenAIEmbeddings') as mock_embeddings, \
         patch('app.rag.Pinecone') as mock_pc, \
         patch('app.rag.PineconeVectorStore', return_value=mock_vectorstore), \
         patch('app.rag.anthropic.Anthropic', return_value=mock_anthropic_client) as mock_anthropic_class:
        
        mock_embeddings.return_value = Mock()
        mock_index = Mock()
        mock_pc.return_value.Index.return_value = mock_index
        
        engine = RAGEngine()
        
        return engine


def test_rag_engine_initialization():
    with patch('app.rag.OpenAIEmbeddings'), \
         patch('app.rag.Pinecone'), \
         patch('app.rag.PineconeVectorStore'), \
         patch('app.rag.anthropic.Anthropic') as mock_anthropic:
        mock_anthropic.return_value = Mock()
        engine = RAGEngine()
        assert engine is not None


def test_format_chunk(rag_engine):
    chunk: RetrievedChunk = {
        "content": "Test content",
        "source": "test.pdf",
        "score": 0.95
    }
    
    formatted = rag_engine._format_chunk(chunk)
    
    assert "[Source: test.pdf]" in formatted
    assert "Test content" in formatted


def test_retrieve_context_empty(rag_engine):
    rag_engine.vectorstore.similarity_search_with_score.return_value = []
    
    chunks = rag_engine.retrieve_context("test query")
    
    assert chunks == []
    rag_engine.vectorstore.similarity_search_with_score.assert_called_once()


def test_retrieve_context_with_results(rag_engine):
    mock_doc1 = Mock()
    mock_doc1.page_content = "First chunk content"
    mock_doc1.metadata = {"source": "doc1.pdf"}
    
    mock_doc2 = Mock()
    mock_doc2.page_content = "Second chunk content"
    mock_doc2.metadata = {"source": "doc2.pdf"}
    
    rag_engine.vectorstore.similarity_search_with_score.return_value = [
        (mock_doc1, 0.95),
        (mock_doc2, 0.85)
    ]
    
    chunks = rag_engine.retrieve_context("test query")
    
    assert len(chunks) == 2
    assert chunks[0]["content"] == "First chunk content"
    assert chunks[0]["source"] == "doc1.pdf"
    assert chunks[0]["score"] == 0.95
    assert chunks[1]["content"] == "Second chunk content"
    assert chunks[1]["source"] == "doc2.pdf"
    assert chunks[1]["score"] == 0.85


def test_retrieve_context_custom_top_k(rag_engine):
    rag_engine.vectorstore.similarity_search_with_score.return_value = []
    
    rag_engine.retrieve_context("test query", top_k=10)
    
    rag_engine.vectorstore.similarity_search_with_score.assert_called_with("test query", k=10)


def test_retrieve_context_missing_metadata(rag_engine):
    mock_doc = Mock()
    mock_doc.page_content = "Content"
    mock_doc.metadata = {}
    
    rag_engine.vectorstore.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
    
    chunks = rag_engine.retrieve_context("test query")
    
    assert chunks[0]["source"] == "unknown"


def test_generate_response(rag_engine):
    chunks: list[RetrievedChunk] = [
        {"content": "Content 1", "source": "doc1.pdf", "score": 0.95},
        {"content": "Content 2", "source": "doc2.pdf", "score": 0.85}
    ]
    
    response = rag_engine.generate_response("What is this?", chunks)
    
    assert response["answer"] == "This is a test answer"
    assert response["sources"] == chunks
    assert response["input_tokens"] == 100
    assert response["output_tokens"] == 50
    assert "model" in response


def test_generate_response_uses_correct_prompts(rag_engine):
    chunks: list[RetrievedChunk] = [
        {"content": "Test content", "source": "test.pdf", "score": 0.9}
    ]
    
    rag_engine.generate_response("What is this?", chunks)
    
    call_args = rag_engine.client.messages.create.call_args
    assert call_args.kwargs["system"] == SYSTEM_PROMPT
    assert "What is this?" in call_args.kwargs["messages"][0]["content"]
    assert "Test content" in call_args.kwargs["messages"][0]["content"]


def test_generate_response_includes_all_chunks_in_context(rag_engine):
    chunks: list[RetrievedChunk] = [
        {"content": "Content A", "source": "a.pdf", "score": 0.9},
        {"content": "Content B", "source": "b.pdf", "score": 0.8},
        {"content": "Content C", "source": "c.pdf", "score": 0.7}
    ]
    
    rag_engine.generate_response("Question", chunks)
    
    call_args = rag_engine.client.messages.create.call_args
    user_message = call_args.kwargs["messages"][0]["content"]
    
    assert "Content A" in user_message
    assert "Content B" in user_message
    assert "Content C" in user_message
    assert "a.pdf" in user_message
    assert "b.pdf" in user_message
    assert "c.pdf" in user_message


def test_query_with_no_context(rag_engine):
    rag_engine.vectorstore.similarity_search_with_score.return_value = []
    
    response = rag_engine.query("What are your hobbies?")
    
    assert "don't have any information" in response["answer"]
    assert response["sources"] == []
    assert response["input_tokens"] == 0
    assert response["output_tokens"] == 0


def test_query_with_context(rag_engine):
    mock_doc = Mock()
    mock_doc.page_content = "I enjoy hiking and coding"
    mock_doc.metadata = {"source": "personal_info.json"}
    
    rag_engine.vectorstore.similarity_search_with_score.return_value = [
        (mock_doc, 0.95)
    ]
    
    response = rag_engine.query("What are your hobbies?")
    
    assert response["answer"] == "This is a test answer"
    assert len(response["sources"]) == 1
    assert response["sources"][0]["content"] == "I enjoy hiking and coding"
    assert response["input_tokens"] == 100
    assert response["output_tokens"] == 50


def test_query_custom_top_k(rag_engine):
    rag_engine.vectorstore.similarity_search_with_score.return_value = []
    
    rag_engine.query("Question", top_k=3)
    
    rag_engine.vectorstore.similarity_search_with_score.assert_called_with("Question", k=3)


def test_query_rag_function():
    with patch('app.rag.get_rag_engine') as mock_get_engine:
        mock_engine = Mock()
        mock_engine.query.return_value = {
            "answer": "Test",
            "sources": [],
            "input_tokens": 10,
            "output_tokens": 5,
            "model": "claude-sonnet-4-5"
        }
        mock_get_engine.return_value = mock_engine
        
        result = query_rag("Test question")
        
        assert result["answer"] == "Test"
        mock_engine.query.assert_called_once_with("Test question", None)


def test_query_rag_function_with_top_k():
    with patch('app.rag.get_rag_engine') as mock_get_engine:
        mock_engine = Mock()
        mock_engine.query.return_value = {
            "answer": "Test",
            "sources": [],
            "input_tokens": 10,
            "output_tokens": 5,
            "model": "claude-sonnet-4-5"
        }
        mock_get_engine.return_value = mock_engine
        
        query_rag("Test question", top_k=7)
        
        mock_engine.query.assert_called_once_with("Test question", 7)


def test_response_type_structure():
    response: RAGResponse = {
        "answer": "Test answer",
        "sources": [
            {"content": "Content", "source": "test.pdf", "score": 0.9}
        ],
        "input_tokens": 100,
        "output_tokens": 50,
        "model": "claude-sonnet-4-5"
    }
    
    assert isinstance(response["answer"], str)
    assert isinstance(response["sources"], list)
    assert isinstance(response["input_tokens"], int)
    assert isinstance(response["output_tokens"], int)
    assert isinstance(response["model"], str)