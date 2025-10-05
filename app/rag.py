from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatAnthropic
from langchain.chains import RetrievalQA

def query_rag(question: str) -> dict:
    # Vector search in Pinecone
    # Call Claude with retrieved context
    # Return response + metadata
    pass