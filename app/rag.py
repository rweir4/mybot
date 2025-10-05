from typing import List, TypedDict, Optional
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pinecone import Pinecone
from app.config import settings
import anthropic


SYSTEM_PROMPT = """You are a helpful AI assistant answering questions about a person based on their personal information, research papers, and other documents. 

Use the provided context to answer questions accurately. If the answer isn't in the context, say so - don't make things up.

When referencing information, you can mention which source it came from if relevant."""

USER_PROMPT_TEMPLATE = """Context from knowledge base:

{context}

---

Question: {query}

Please provide a helpful and accurate answer based on the context above."""


class RetrievedChunk(TypedDict):
    content: str
    source: str
    score: float


class RAGResponse(TypedDict):
    answer: str
    sources: List[RetrievedChunk]
    input_tokens: int
    output_tokens: int
    model: str


class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index_name)
        
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"
        )
        
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    
    def _format_chunk(self, chunk: RetrievedChunk) -> str:
        return f"[Source: {chunk['source']}]\n{chunk['content']}"
    
    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k or settings.retrieval_top_k
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        chunks: List[RetrievedChunk] = []
        for doc, score in results:
            chunks.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score)
            })
        
        return chunks
    
    def generate_response(self, query: str, context_chunks: List[RetrievedChunk]) -> RAGResponse:
        context = "\n\n".join([self._format_chunk(chunk) for chunk in context_chunks])
        
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)
        
        response = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_output_tokens,
            temperature=settings.claude_temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        answer = response.content[0].text
        
        return {
            "answer": answer,
            "sources": context_chunks,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": settings.claude_model
        }
    
    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        chunks = self.retrieve_context(question, top_k)
        
        if not chunks:
            return {
                "answer": "I don't have any information in my knowledge base to answer that question.",
                "sources": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "model": settings.claude_model
            }
        
        return self.generate_response(question, chunks)

_rag_engine = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def query_rag(question: str, top_k: Optional[int] = None) -> RAGResponse:
    return get_rag_engine().query(question, top_k)