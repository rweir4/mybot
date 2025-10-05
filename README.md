# Personal Chatbot API

AI-powered chatbot API with knowledge about personal information, research papers, and interests. Built with FastAPI, Claude (Anthropic), and Pinecone for semantic search.

## Features

- **RAG (Retrieval-Augmented Generation)** - Semantic search over personal documents and research papers
- **Rate Limiting** - Configurable hourly limits and daily cost caps to control API usage
- **Usage Tracking** - Detailed logging of requests, tokens, and costs
- **CORS Enabled** - Ready for frontend integration
- **Type-Safe** - Full TypeScript-style typing with Pydantic
- **Idempotent Ingestion** - Update knowledge base without duplicates
- **Emergency Kill Switch** - Disable API via environment variable

## Tech Stack

- **FastAPI** - Modern Python web framework
- **Claude (Anthropic)** - LLM for generating responses
- **Pinecone** - Vector database for semantic search
- **LangChain** - Document processing and RAG orchestration
- **OpenAI** - Embedding generation
- **Pydantic** - Data validation and settings management

## Setup

### Prerequisites

- Python 3.11 (not 3.13 - compatibility issues with some dependencies)
- API keys for:
  - Anthropic Claude
  - Pinecone
  - OpenAI

### Installation

```bash
# Clone the repository
git clone git@github.com:rweir4/mybot.git
cd personal-chatbot-api

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file from template
cp .env.example .env
```

### Configuration

Edit `.env` with your API keys and settings:

```bash
# API Keys (required)
ANTHROPIC_API_KEY=sk-ant-your-key-here
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=sk-your-openai-key

# Pinecone Configuration
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=personal-chatbot

# Rate Limiting
RATE_LIMIT_PER_HOUR=30
MAX_OUTPUT_TOKENS=4000
MAX_DAILY_COST=5.0

# Emergency Kill Switch
API_ENABLED=true
```

### Prepare Your Data

1. **Create personal info file:**

```bash
# Create data directory
mkdir -p data/papers

# Create personal info JSON
cat > data/personal_info.json << 'EOF'
{
  "name": "Your Name",
  "bio": "Brief bio",
  "interests": {
    "hobbies": ["hobby1", "hobby2"],
    "research_areas": ["area1", "area2"]
  }
}
EOF
```

2. **Add research papers:**

```bash
# Copy your PDF papers
cp /path/to/your/papers/*.pdf data/papers/
```

### Ingest Data

Run the ingestion script to populate Pinecone:

```bash
python3 -m scripts.ingest
```

This will:
- Load your personal info and papers
- Chunk documents into searchable pieces
- Generate embeddings using OpenAI
- Upload to Pinecone with idempotent upsert

You can re-run this anytime to update your knowledge base.

## Running the API

### Local Development

```bash
# Start the server
uvicorn app.main:app --reload

# API will be available at:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
# - OpenAPI spec: http://localhost:8000/openapi.json
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /chat

Ask a question and get an AI-generated response with sources.

**Request:**
```json
{
  "message": "What are your research interests?",
  "top_k": 5  // optional, number of context chunks to retrieve
}
```

**Response:**
```json
{
  "answer": "Based on my research...",
  "sources": [
    {
      "content": "Excerpt from source...",
      "source": "paper1.pdf",
      "relevance_score": 0.95
    }
  ],
  "tokens_used": 1234,
  "estimated_cost": 0.0185
}
```

### GET /health

Check API health and configuration status.

**Response:**
```json
{
  "status": "healthy",
  "api_enabled": true,
  "config_valid": true
}
```

### GET /stats

Get usage statistics and rate limit status.

**Response:**
```json
{
  "rate_limit": {
    "hourly_stats": {
      "requests_used": 15,
      "requests_limit": 30,
      "requests_remaining": 15
    },
    "daily_stats": {
      "tokens_used": 45000,
      "estimated_cost": 0.23,
      "cost_limit": 5.0
    }
  },
  "usage": {
    "total_requests": 120,
    "successful_requests": 118,
    "total_cost_usd": 2.45
  }
}
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_main.py -v

# Run specific test
pytest tests/test_main.py::test_chat_endpoint_success -v
```

## Deployment

### Railway

1. Create a new Railway project
2. Connect your GitHub repository
3. Add environment variables in Railway dashboard (all values from `.env`)
4. Railway will auto-deploy on push to main branch

The API will be available at: `https://your-project.railway.app`

### Environment Variables on Railway

Make sure to set all required environment variables:
- `ANTHROPIC_API_KEY`
- `PINECONE_API_KEY`
- `OPENAI_API_KEY`
- `PINECONE_ENVIRONMENT`
- `PINECONE_INDEX_NAME`
- `RATE_LIMIT_PER_HOUR`
- `MAX_DAILY_COST`
- `API_ENABLED`

## Project Structure

```
personal-chatbot-api/
├── app/
│   ├── config.py          # Configuration management
│   ├── rate_limiter.py    # Rate limiting logic
│   ├── logger.py          # Usage logging
│   ├── rag.py             # RAG implementation
│   └── main.py            # FastAPI application
├── scripts/
│   └── ingest.py          # Data ingestion script
├── tests/
│   ├── test_config.py
│   ├── test_rate_limiter.py
│   ├── test_logger.py
│   ├── test_rag.py
│   ├── test_ingest.py
│   └── test_main.py
├── data/
│   ├── personal_info.json # Your personal information
│   └── papers/            # PDF research papers
├── requirements.txt       # Python dependencies
├── pytest.ini            # Pytest configuration
├── .env.example          # Environment variables template
└── README.md
```

## Emergency Controls

### Kill Switch

Disable the API immediately:

```bash
# In Railway dashboard or .env
API_ENABLED=false
```

All requests will return 503 Service Unavailable.

### Rate Limit Adjustment

Update limits on the fly:

```bash
# Increase hourly limit
RATE_LIMIT_PER_HOUR=100

# Increase daily cost cap
MAX_DAILY_COST=10.0
```

## Cost Management

The API tracks costs automatically:

- **Embeddings** (one-time during ingestion): ~$0.02 per 1M tokens
- **Claude API** (per query):
  - Input: $3 per 1M tokens
  - Output: $15 per 1M tokens
- **Daily cap**: Configurable via `MAX_DAILY_COST`

Expected costs for typical usage:
- Ingestion (one-time): <$0.01
- 100 queries/day: ~$0.50-2.00/day depending on complexity

## Updating Knowledge Base

To update your information:

1. Edit `data/personal_info.json` or add PDFs to `data/papers/`
2. Re-run ingestion:

```bash
python3 -m scripts.ingest
```

The script uses idempotent upsert - only changed content is re-processed.

## Troubleshooting

### Python 3.13 Compatibility Issues

Use Python 3.11:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### ModuleNotFoundError

Make sure virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Rate Limit Errors

Check your current usage:

```bash
curl http://localhost:8000/stats
```

Adjust `RATE_LIMIT_PER_HOUR` if needed.

### Pinecone Index Creation Fails

Verify your Pinecone environment matches your account region:

```bash
# Check Pinecone dashboard for your environment
# Update in .env:
PINECONE_ENVIRONMENT=us-east-1-aws  # or your region
```

## License

MIT

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request