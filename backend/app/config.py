from pydantic_settings import BaseSettings
from typing import List, Dict


class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # CORS (comma-separated string, will be split)
    CORS_ORIGINS: str = "http://localhost:3000"
    
    # AI Provider API Keys
    GOOGLE_GEMINI_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    
    # LangChain/LangGraph Settings
    DEFAULT_MODEL: str = "gemini"  # gemini or deepseek
    
    # Tool Settings (placeholders - implementation deferred)
    TAVILY_API_KEY: str = ""  # Placeholder for Tavily API key
    MCP_SERVERS: Dict[str, Dict] = {}  # Placeholder for MCP server configs
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS origins to list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    def load_enabled_mcp_servers(self) -> List[str]:
        """Load enabled MCP servers from config. Placeholder - implementation deferred."""
        # Future implementation: return list of enabled MCP server names
        return []
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

