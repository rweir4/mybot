from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from app.config import settings, validate_config, get_config_summary
from app.rate_limiter import rate_limiter
from app.logger import usage_logger
from app.rag import query_rag, RAGResponse


app = FastAPI(
    title="Personal Chatbot API",
    description="AI-powered chatbot with knowledge about a person's background, research, and interests",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of context chunks to retrieve")


class ChatResponse(BaseModel):
    answer: str
    sources: list
    tokens_used: int
    estimated_cost: float


class HealthResponse(BaseModel):
    status: str
    api_enabled: bool
    config_valid: bool


class StatsResponse(BaseModel):
    rate_limit: dict
    usage: dict


@app.on_event("startup")
async def startup_event():
    try:
        validate_config()
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Personal Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        validate_config()
        config_valid = True
    except ValueError:
        config_valid = False
    
    return {
        "status": "healthy" if settings.api_enabled else "disabled",
        "api_enabled": settings.api_enabled,
        "config_valid": config_valid
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        usage_stats = rate_limiter.check_rate_limit()
    except HTTPException as e:
        usage_logger.log_request(
            endpoint="/chat",
            input_tokens=0,
            output_tokens=0,
            estimated_cost=0.0,
            success=False,
            error=str(e.detail)
        )
        raise
    
    try:
        rag_response: RAGResponse = query_rag(request.message, request.top_k)
        
        input_tokens = rag_response["input_tokens"]
        output_tokens = rag_response["output_tokens"]
        
        usage_record = rate_limiter.record_usage(input_tokens, output_tokens)
        estimated_cost = usage_record["estimated_cost"]
        
        usage_logger.log_request(
            endpoint="/chat",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            success=True,
            metadata={
                "message_length": len(request.message),
                "sources_count": len(rag_response["sources"]),
                "top_k": request.top_k or settings.retrieval_top_k
            }
        )
        
        return {
            "answer": rag_response["answer"],
            "sources": [
                {
                    "content": source["content"][:200] + "..." if len(source["content"]) > 200 else source["content"],
                    "source": source["source"],
                    "relevance_score": round(source["score"], 3)
                }
                for source in rag_response["sources"]
            ],
            "tokens_used": input_tokens + output_tokens,
            "estimated_cost": estimated_cost
        }
    
    except Exception as e:
        usage_logger.log_request(
            endpoint="/chat",
            input_tokens=0,
            output_tokens=0,
            estimated_cost=0.0,
            success=False,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def stats():
    return {
        "rate_limit": rate_limiter.get_stats(),
        "usage": usage_logger.get_stats()
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)