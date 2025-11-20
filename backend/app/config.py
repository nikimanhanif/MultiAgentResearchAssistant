from pydantic_settings import BaseSettings
from typing import List


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
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS origins to list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

