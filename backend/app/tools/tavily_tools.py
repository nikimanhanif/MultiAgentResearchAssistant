"""
Tavily web search tool configuration.

Provides web search capabilities via langchain-tavily integration,
using native LangChain Tavily tools (TavilySearch, TavilyExtract) for research.
"""

import logging
from typing import List

from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch, TavilyExtract

from app.config import settings

logger = logging.getLogger(__name__)


def get_tavily_tools() -> List[BaseTool]:
    """
    Get configured Tavily tools for web search and extraction.
    
    Returns a list containing:
    - TavilySearch: Web search with advanced depth.
    - TavilyExtract: Content extraction from URLs.
    
    Configuration:
    - max_results: 5 (balanced between quality and token usage).
    - search_depth: "advanced" (comprehensive search).
    
    Returns:
        List[BaseTool]: List of configured Tavily tools.
        
    Raises:
        ValueError: If TAVILY_API_KEY is not configured.
    """
    if not settings.TAVILY_API_KEY:
        error_msg = "TAVILY_API_KEY not configured in environment"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        tavily_search = TavilySearch(
            api_key=settings.TAVILY_API_KEY,
            max_results=5,
            search_depth="advanced"
        )
        
        tavily_extract = TavilyExtract(
            api_key=settings.TAVILY_API_KEY,
            extract_depth="basic"
        )
        
        tools = [tavily_search, tavily_extract]
        logger.info(f"Successfully initialized {len(tools)} Tavily tools")
        return tools
        
    except Exception as e:
        error_msg = f"Failed to initialize Tavily tools: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
