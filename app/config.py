from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "personal-chatbot"
    rate_limit_per_hour: int = 30
    max_output_tokens: int = 4000
    api_enabled: bool = True
    max_daily_cost: float = 5.0
    
    class Config:
        env_file = ".env"

settings = Settings()