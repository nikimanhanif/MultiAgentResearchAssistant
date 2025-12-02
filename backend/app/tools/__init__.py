"""
Tools module for research agent integrations.

This module provides tool integrations using native LangChain packages:
- Tavily: Web search and extraction.
- MCP: Multi-tool Web Server integrations with automatic discovery.
- Registry: Centralized tool loading and safety wrapping.
"""

from app.tools.tavily_tools import get_tavily_tools
from app.tools.mcp_tools import get_mcp_client, load_mcp_tools
from app.tools.tool_registry import get_research_tools

__all__ = ["get_tavily_tools", "get_mcp_client", "load_mcp_tools", "get_research_tools"]
