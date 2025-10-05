import json
from pathlib import Path
from typing import List, Set, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from app.config import settings


def load_personal_info(file_path: str = "data/personal_info.json") -> List:
    if not Path(file_path).exists():
        print(f"⚠️  Personal info file not found: {file_path}")
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    text_content = json.dumps(data, indent=2)
    
    from langchain.schema import Document
    return [Document(
        page_content=text_content,
        metadata={"source": "personal_info.json", "type": "personal_data"}
    )]


def load_papers(papers_dir: str = "data/papers") -> List:
    papers_path = Path(papers_dir)
    
    if not papers_path.exists():
        print(f"⚠️  Papers directory not found: {papers_dir}")
        return []
    
    pdf_files = list(papers_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {papers_dir}")
        return []
    
    documents = []
    for pdf_file in pdf_files:
        print(f"📄 Loading {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = pdf_file.name
            doc.metadata["type"] = "research_paper"
        
        documents.extend(docs)
    
    print(f"✓ Loaded {len(documents)} pages from {len(pdf_files)} papers")
    return documents


def chunk_documents(documents: List, chunk_size: int = None, chunk_overlap: int = None) -> List:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    return chunks


def create_or_get_index(index_name: str):
    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}...")
        
        # Extract region from environment (e.g., "us-east-1-aws" -> "us-east-1")
        region = settings.pinecone_environment.rsplit('-', 1)[0]
        
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region)
        )
        print(f"✓ Index created: {index_name}")
    else:
        print(f"✓ Using existing index: {index_name}")
    
    return pc.Index(index_name)


def get_existing_ids_for_source(index, source: str) -> Set[str]:
    try:
        results = index.query(
            vector=[0.0] * settings.embedding_dimensions,
            filter={"source": {"$eq": source}},
            top_k=10000,
            include_metadata=False
        )
        
        existing_ids = {match.id for match in results.matches}
        return existing_ids
    except Exception as e:
        print(f"⚠️  Could not fetch existing IDs for {source}: {e}")
        return set()


def ingest_source_to_pinecone(
    index,
    embeddings,
    chunks: List,
    source_name: str
) -> Tuple[int, int]:
    if not chunks:
        return 0, 0
    
    print(f"  📤 Upserting {len(chunks)} chunks for {source_name}...")
    
    chunk_ids = []
    vectors_to_upsert = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_name}-chunk-{i}"
        chunk_ids.append(chunk_id)
        
        embedding = embeddings.embed_query(chunk.page_content)
        
        vectors_to_upsert.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
                "type": chunk.metadata.get("type", "unknown"),
                "chunk_index": i
            }
        })
    
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"  ✓ Upserted {len(chunk_ids)} chunks")
    
    print(f"  🔍 Checking for orphaned chunks...")
    existing_ids = get_existing_ids_for_source(index, source_name)
    new_ids = set(chunk_ids)
    orphan_ids = existing_ids - new_ids
    
    if orphan_ids:
        print(f"  🗑️  Deleting {len(orphan_ids)} orphaned chunks...")
        index.delete(ids=list(orphan_ids))
        deleted_count = len(orphan_ids)
    else:
        print(f"  ✓ No orphaned chunks found")
        deleted_count = 0
    
    return len(chunk_ids), deleted_count


def ingest_to_pinecone(documents_by_source: dict):
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key
    )
    
    index = create_or_get_index(settings.pinecone_index_name)
    
    total_upserted = 0
    total_deleted = 0
    
    for source_name, documents in documents_by_source.items():
        print(f"\n🔄 Processing {source_name}...")
        chunks = chunk_documents(documents)
        
        upserted, deleted = ingest_source_to_pinecone(
            index=index,
            embeddings=embeddings,
            chunks=chunks,
            source_name=source_name
        )
        
        total_upserted += upserted
        total_deleted += deleted
    
    print(f"\n✓ Total chunks upserted: {total_upserted}")
    if total_deleted > 0:
        print(f"✓ Total orphaned chunks deleted: {total_deleted}")


def main():
    print("=" * 60)
    print("Personal Chatbot - Data Ingestion")
    print("=" * 60)
    
    documents_by_source = {}
    
    print("\n1. Loading personal information...")
    personal_docs = load_personal_info()
    if personal_docs:
        documents_by_source["personal_info.json"] = personal_docs
    
    print("\n2. Loading research papers...")
    paper_docs = load_papers()
    
    papers_by_file = {}
    for doc in paper_docs:
        source = doc.metadata["source"]
        if source not in papers_by_file:
            papers_by_file[source] = []
        papers_by_file[source].append(doc)
    
    documents_by_source.update(papers_by_file)
    
    if not documents_by_source:
        print("\n❌ No documents found to ingest!")
        print("Please add:")
        print("  - data/personal_info.json")
        print("  - PDF files in data/papers/")
        return
    
    total_docs = sum(len(docs) for docs in documents_by_source.values())
    print(f"\n✓ Total documents loaded: {total_docs} from {len(documents_by_source)} sources")
    
    print("\n3. Uploading to Pinecone with upsert...")
    ingest_to_pinecone(documents_by_source)
    
    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print("=" * 60)
    print("\nYour knowledge base has been updated.")
    print("You can now run the API and start asking questions!")
    print("\n💡 Tip: Re-run this script anytime to update your data.")


if __name__ == "__main__":
    main()