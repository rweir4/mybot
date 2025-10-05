from fastapi import FastAPI
from app.rag import query_rag
from app.rate_limiter import check_rate_limit
from app.config import settings
from app.logger import log_usage

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    check_rate_limit()
    response = query_rag(request.message)
    log_usage(tokens=response.tokens, cost=response.cost)
    return response

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/stats")
async def stats():
    return get_usage_stats()