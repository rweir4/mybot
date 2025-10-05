import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

import sys
sys.modules['langchain_openai'] = Mock()
sys.modules['langchain.document_loaders'] = Mock()
sys.modules['pinecone'] = Mock()

from scripts.ingest import (
    load_personal_info,
    load_papers,
    chunk_documents,
    get_existing_ids_for_source,
    ingest_source_to_pinecone
)


@pytest.fixture
def temp_personal_info():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        data = {
            "name": "Test User",
            "hobbies": ["coding", "reading"],
            "bio": "A test user"
        }
        json.dump(data, f)
        temp_path = f.name
    
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_papers_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_load_personal_info_success(temp_personal_info):
    docs = load_personal_info(temp_personal_info)
    
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "personal_info.json"
    assert docs[0].metadata["type"] == "personal_data"
    assert "Test User" in docs[0].page_content


def test_load_personal_info_file_not_found():
    docs = load_personal_info("nonexistent.json")
    assert docs == []


def test_load_personal_info_contains_all_data(temp_personal_info):
    docs = load_personal_info(temp_personal_info)
    content = docs[0].page_content
    
    assert "Test User" in content
    assert "coding" in content
    assert "reading" in content
    assert "A test user" in content


def test_load_papers_no_directory():
    docs = load_papers("nonexistent_dir")
    assert docs == []


def test_load_papers_empty_directory(temp_papers_dir):
    docs = load_papers(temp_papers_dir)
    assert docs == []


def test_load_papers_with_pdfs(temp_papers_dir):
    with patch('scripts.ingest.PyPDFLoader') as mock_loader:
        Path(temp_papers_dir, "paper1.pdf").touch()
        Path(temp_papers_dir, "paper2.pdf").touch()
        
        mock_doc1 = Document(page_content="Page 1 content", metadata={})
        mock_doc2 = Document(page_content="Page 2 content", metadata={})
        
        mock_loader.return_value.load.return_value = [mock_doc1, mock_doc2]
        
        docs = load_papers(temp_papers_dir)
        
        assert len(docs) == 4
        assert mock_loader.call_count == 2


def test_chunk_documents_empty_list():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_documents_creates_chunks():
    with patch('scripts.ingest.RecursiveCharacterTextSplitter') as mock_splitter:
        mock_instance = Mock()
        mock_instance.split_documents.return_value = [
            Document(page_content="chunk1", metadata={}),
            Document(page_content="chunk2", metadata={})
        ]
        mock_splitter.return_value = mock_instance
        
        docs = [Document(page_content="test content", metadata={})]
        chunks = chunk_documents(docs)
        
        assert len(chunks) == 2
        mock_splitter.assert_called_once()


def test_chunk_documents_uses_config_settings():
    with patch('scripts.ingest.RecursiveCharacterTextSplitter') as mock_splitter, \
         patch('scripts.ingest.settings') as mock_settings:
        
        mock_settings.chunk_size = 500
        mock_settings.chunk_overlap = 100
        mock_instance = Mock()
        mock_instance.split_documents.return_value = []
        mock_splitter.return_value = mock_instance
        
        chunk_documents([])
        
        call_kwargs = mock_splitter.call_args.kwargs
        assert call_kwargs['chunk_size'] == 500
        assert call_kwargs['chunk_overlap'] == 100


def test_chunk_documents_custom_params():
    with patch('scripts.ingest.RecursiveCharacterTextSplitter') as mock_splitter:
        mock_instance = Mock()
        mock_instance.split_documents.return_value = []
        mock_splitter.return_value = mock_instance
        
        chunk_documents([], chunk_size=200, chunk_overlap=50)
        
        call_kwargs = mock_splitter.call_args.kwargs
        assert call_kwargs['chunk_size'] == 200
        assert call_kwargs['chunk_overlap'] == 50


def test_get_existing_ids_for_source():
    mock_index = Mock()
    mock_match1 = Mock()
    mock_match1.id = "source1-chunk-0"
    mock_match2 = Mock()
    mock_match2.id = "source1-chunk-1"
    
    mock_index.query.return_value = Mock(matches=[mock_match1, mock_match2])
    
    with patch('scripts.ingest.settings') as mock_settings:
        mock_settings.embedding_dimensions = 1536
        
        ids = get_existing_ids_for_source(mock_index, "source1")
        
        assert ids == {"source1-chunk-0", "source1-chunk-1"}
        mock_index.query.assert_called_once()


def test_get_existing_ids_handles_errors():
    mock_index = Mock()
    mock_index.query.side_effect = Exception("API Error")
    
    with patch('scripts.ingest.settings') as mock_settings:
        mock_settings.embedding_dimensions = 1536
        
        ids = get_existing_ids_for_source(mock_index, "source1")
        
        assert ids == set()


def test_ingest_source_to_pinecone_empty_chunks():
    mock_index = Mock()
    mock_embeddings = Mock()
    
    upserted, deleted = ingest_source_to_pinecone(
        index=mock_index,
        embeddings=mock_embeddings,
        chunks=[],
        source_name="test"
    )
    
    assert upserted == 0
    assert deleted == 0
    mock_index.upsert.assert_not_called()


def test_ingest_source_to_pinecone_upserts_chunks():
    mock_index = Mock()
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1536
    
    mock_index.query.return_value = Mock(matches=[])
    
    chunks = [
        Document(page_content="chunk 1", metadata={"source": "test.pdf", "type": "paper"}),
        Document(page_content="chunk 2", metadata={"source": "test.pdf", "type": "paper"})
    ]
    
    upserted, deleted = ingest_source_to_pinecone(
        index=mock_index,
        embeddings=mock_embeddings,
        chunks=chunks,
        source_name="test.pdf"
    )
    
    assert upserted == 2
    assert deleted == 0
    assert mock_index.upsert.call_count == 1


def test_ingest_source_to_pinecone_deletes_orphans():
    mock_index = Mock()
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1536
    
    mock_match1 = Mock()
    mock_match1.id = "test.pdf-chunk-0"
    mock_match2 = Mock()
    mock_match2.id = "test.pdf-chunk-1"
    mock_match3 = Mock()
    mock_match3.id = "test.pdf-chunk-2"
    
    mock_index.query.return_value = Mock(matches=[mock_match1, mock_match2, mock_match3])
    
    chunks = [
        Document(page_content="chunk 1", metadata={"source": "test.pdf", "type": "paper"})
    ]
    
    with patch('scripts.ingest.settings') as mock_settings:
        mock_settings.embedding_dimensions = 1536
        
        upserted, deleted = ingest_source_to_pinecone(
            index=mock_index,
            embeddings=mock_embeddings,
            chunks=chunks,
            source_name="test.pdf"
        )
    
    assert upserted == 1
    assert deleted == 2
    mock_index.delete.assert_called_once()
    deleted_ids = mock_index.delete.call_args.kwargs['ids']
    assert set(deleted_ids) == {"test.pdf-chunk-1", "test.pdf-chunk-2"}


def test_ingest_source_to_pinecone_generates_correct_ids():
    mock_index = Mock()
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1536
    
    mock_index.query.return_value = Mock(matches=[])
    
    chunks = [
        Document(page_content="chunk 1", metadata={"source": "test.pdf", "type": "paper"}),
        Document(page_content="chunk 2", metadata={"source": "test.pdf", "type": "paper"}),
        Document(page_content="chunk 3", metadata={"source": "test.pdf", "type": "paper"})
    ]
    
    ingest_source_to_pinecone(
        index=mock_index,
        embeddings=mock_embeddings,
        chunks=chunks,
        source_name="test.pdf"
    )
    
    upsert_call = mock_index.upsert.call_args
    vectors = upsert_call.kwargs['vectors']
    
    assert len(vectors) == 3
    assert vectors[0]['id'] == "test.pdf-chunk-0"
    assert vectors[1]['id'] == "test.pdf-chunk-1"
    assert vectors[2]['id'] == "test.pdf-chunk-2"


def test_ingest_source_to_pinecone_includes_metadata():
    mock_index = Mock()
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1536
    
    mock_index.query.return_value = Mock(matches=[])
    
    chunks = [
        Document(
            page_content="test content",
            metadata={"source": "test.pdf", "type": "research_paper"}
        )
    ]
    
    ingest_source_to_pinecone(
        index=mock_index,
        embeddings=mock_embeddings,
        chunks=chunks,
        source_name="test.pdf"
    )
    
    vectors = mock_index.upsert.call_args.kwargs['vectors']
    metadata = vectors[0]['metadata']
    
    assert metadata['text'] == "test content"
    assert metadata['source'] == "test.pdf"
    assert metadata['type'] == "research_paper"
    assert metadata['chunk_index'] == 0


def test_ingest_source_to_pinecone_batches_large_uploads():
    mock_index = Mock()
    mock_embeddings = Mock()
    mock_embeddings.embed_query.return_value = [0.1] * 1536
    
    mock_index.query.return_value = Mock(matches=[])
    
    chunks = [
        Document(page_content=f"chunk {i}", metadata={"source": "test.pdf", "type": "paper"})
        for i in range(250)
    ]
    
    ingest_source_to_pinecone(
        index=mock_index,
        embeddings=mock_embeddings,
        chunks=chunks,
        source_name="test.pdf"
    )
    
    assert mock_index.upsert.call_count == 3