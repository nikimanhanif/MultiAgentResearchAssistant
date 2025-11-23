"""Application configuration using Pydantic settings.

Configuration Architecture (Phase 2.5):
- Environment variables loaded from .env file
- Tool configurations moved to dedicated modules (Phase 7):
  - MCP server configs → app/tools/mcp_tools.py
  - Research sources → app/utils/research_sources.py
- LangGraph persistence config added in Phase 6.3 (PostgresSaver)

Phase References:
- Phase 6.3: Add DATABASE_URL for conversation persistence
- Phase 7.1: TAVILY_API_KEY used by tavily_tools.py
- Phase 7.3: CORE_API_KEY for Scientific Paper Harvester MCP
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict
from pathlib import Path


class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"
    
    # Research Pipeline LLM (DeepSeek)
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    
    # Evaluation Framework LLM (OpenAI)
    OPENAI_API_KEY: str = ""
    OPENAI_EVAL_MODEL: str = "gpt-5-nano"
    
    # Deprecated API Keys (kept for backwards compatibility, not used in current system)
    GOOGLE_GEMINI_API_KEY: str = ""  # DEPRECATED: Gemini no longer used (DeepSeek-only for research)
    
    # Tool Settings
    TAVILY_API_KEY: str = ""  # Phase 7.1 - Tavily web search
    CORE_API_KEY: str = ""    # Phase 7.3 - CORE academic database (via Scientific Paper Harvester)
    
    # Evaluation Framework Settings 
    LANGSMITH_API_KEY: str = ""  # LangSmith API key for trace observability
    LANGSMITH_TRACING: str = "true"  # Enable LangSmith tracing (default: true)
    LANGSMITH_PROJECT: str = "multi-agent-research-assistant-eval"  # LangSmith project name
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"  # LangSmith API endpoint
    DEEPEVAL_TELEMETRY_OPT_OUT: str = "true"  # Opt out of DeepEval telemetry (default: true)
    
    # NOTE: Phase 7.2 - MCP server configs moved to app/tools/mcp_tools.py (server_configs dict)
    # NOTE: Phase 6.2 - Research sources moved to app/utils/research_sources.py (CURATED_RESEARCH_SOURCES)
    
    # Database Settings (Phase 6.3 - for conversation persistence with PostgresSaver)
    DATABASE_URL: str = ""  # PostgreSQL connection string for LangGraph checkpointing
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS origins to list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    def load_enabled_mcp_servers(self) -> List[str]:
        """Load enabled MCP servers from config. Placeholder - implementation deferred."""
        # Future implementation: return list of enabled MCP server names
        return []
    
    model_config = SettingsConfigDict(
        # Use absolute path to .env file relative to this config file
        env_file=str(Path(__file__).parent.parent / ".env"),
        case_sensitive=True
    )


settings = Settings()

