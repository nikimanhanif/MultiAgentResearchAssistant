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
    DEEPSEEK_MODEL: str = "deepseek-chat"  # For scoping agent and sub-agents
    DEEPSEEK_REASONER_MODEL: str = "deepseek-reasoner"  # For supervisor and report agents
    
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


# LLM Factory Functions

def get_deepseek_chat(
    temperature: float = 0.7,
    model: str | None = None
) -> "ChatDeepSeek":
    """Get configured DeepSeek chat model instance.
    
    This model is optimized for:
    - Scoping Agent: Multi-turn clarification conversations
    - Sub-Agents: Focused research tasks with tool usage
    
    Args:
        temperature: Temperature for LLM generation (0.0-1.0)
        model: Optional model override (defaults to settings.DEEPSEEK_MODEL)
        
    Returns:
        Configured ChatDeepSeek instance for chat tasks
        
    Raises:
        ValueError: If DEEPSEEK_API_KEY not configured
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
    """Get configured DeepSeek reasoner model instance.
    
    This model is optimized for:
    - Supervisor Agent: Gap analysis, task generation, findings aggregation
    - Report Agent: Comprehensive report generation with structured reasoning
    
    Features:
    - Larger output token limit (32K default, 64K max vs 4K/8K for chat)
    - Better for complex reasoning and long-form generation
    - Same context window (128K) as chat model
    
    Args:
        temperature: Temperature for LLM generation (0.0-1.0)
                    Lower default (0.5) for more consistent reasoning
        model: Optional model override (defaults to settings.DEEPSEEK_REASONER_MODEL)
        
    Returns:
        Configured ChatDeepSeek instance for reasoning tasks
        
    Raises:
        ValueError: If DEEPSEEK_API_KEY not configured
    """
    from langchain_deepseek import ChatDeepSeek
    
    if not settings.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not configured")
    
    return ChatDeepSeek(
        model=model or settings.DEEPSEEK_REASONER_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=temperature,
    )

