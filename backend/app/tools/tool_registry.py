"""
Central tool registry for research pipeline.

Combines all research tools (Tavily + Academic) into a unified registry.
Handles dynamic tool loading. Safety wrappers are now handled by ToolSafetyMiddleware.
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import List, Optional, AsyncGenerator

from langchain_core.tools import BaseTool

from app.tools.tavily_tools import get_tavily_tools
from app.tools.academic_tools import get_academic_tools

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_research_tools(enabled_mcp_servers: Optional[List[str]] = None) -> AsyncGenerator[List[BaseTool], None]:
    """
    Get all configured research tools (Tavily + Academic) within a context manager.
    
    Note: The context manager pattern is preserved for backwards compatibility
    with existing sub_agent.py code.
    
    Args:
        enabled_mcp_servers: DEPRECATED - Kept for backward compatibility.
                           Academic tools are always loaded regardless of this parameter.
                           
    Yields:
        List[BaseTool]: List of configured tools.
    """
    tools: List[BaseTool] = []
    
    # 1. Load Tavily Tools
    if os.getenv("DISABLE_TAVILY", "").lower() == "true":
        logger.info("Tavily tools disabled via environment variable")
    else:
        try:
            tavily_tools = get_tavily_tools()
            tools.extend(tavily_tools)
            logger.info(f"Loaded {len(tavily_tools)} Tavily tool(s)")
        except Exception as e:
            logger.error(f"Failed to load Tavily tools: {e}")
    
    # 2. Load Academic Tools
    try:
        academic_tools = get_academic_tools()
        tools.extend(academic_tools)
        logger.info(f"Loaded {len(academic_tools)} academic tool(s): {[t.name for t in academic_tools]}")
    except Exception as e:
        logger.error(f"Failed to load academic tools: {e}")
    
    logger.info(f"Total research tools available: {len(tools)}")
    yield tools
