"""
Application configuration using Pydantic settings.

Manages environment variables, API keys, and system-wide settings.
Centralizes configuration for:
- API and CORS settings
- LLM providers (DeepSeek, OpenAI)
- External tools (Tavily, MCP)
- Observability (LangSmith)
- Persistence (Postgres)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    CORS_ORIGINS: str = "http://localhost:3000"
    
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_REASONER_MODEL: str = "deepseek-reasoner"
    
    OPENAI_API_KEY: str = ""
    OPENAI_EVAL_MODEL: str = "gpt-5-nano"
    
    # Deprecated API Keys (kept for backwards compatibility)
    GOOGLE_GEMINI_API_KEY: str = ""
    
    TAVILY_API_KEY: str = ""
    CORE_API_KEY: str = ""
    
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_PROJECT: str = "multi-agent-research-assistant-eval"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    DEEPEVAL_TELEMETRY_OPT_OUT: str = "true"
    
    # Database Settings (Deprecated)
    DATABASE_URL: str = "" 
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS origins to a list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        case_sensitive=True
    )


settings = Settings()


# LLM Factory Functions

def get_deepseek_chat(
    temperature: float = 0.7,
    model: str | None = None
) -> "ChatDeepSeek":
    """
    Get a configured DeepSeek chat model instance.
    
    Used for scoping conversations and sub-agent research tasks.
    
    Args:
        temperature: Controls randomness (0.0-1.0).
        model: Optional model override.
        
    Returns:
        ChatDeepSeek: Configured instance for chat tasks.
        
    Raises:
        ValueError: If DEEPSEEK_API_KEY is missing.
    """
    from langchain_deepseek import ChatDeepSeek
    
    if not settings.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not configured")
    
    return ChatDeepSeek(
        model=model or settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=temperature,
    )


def get_deepseek_reasoner(
    temperature: float = 0.5,
    model: str | None = None
) -> "ChatDeepSeek":
    """
    Get a configured DeepSeek reasoner model instance.
    
    Used for supervisor gap analysis and report generation.
    Lower default temperature ensures consistent reasoning.
    
    Args:
        temperature: Controls randomness (0.0-1.0).
        model: Optional model override.
        
    Returns:
        ChatDeepSeek: Configured instance for reasoning tasks.
        
    Raises:
        ValueError: If DEEPSEEK_API_KEY is missing.
    """
    from langchain_deepseek import ChatDeepSeek
    
    if not settings.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not configured")
    
    return ChatDeepSeek(
        model=model or settings.DEEPSEEK_REASONER_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=temperature,
    )
