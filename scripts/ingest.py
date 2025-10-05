from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

def ingest_documents():
    # Load PDFs from data/papers/
    # Load personal_info.json
    # Chunk everything
    # Generate embeddings
    # Upload to Pinecone
    pass

if __name__ == "__main__":
    ingest_documents()
    print("Ingestion complete!")