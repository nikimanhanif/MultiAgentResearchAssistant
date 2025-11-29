"""Tools module for research agent integrations.

This module provides tool integrations using native LangChain packages:
- langchain-mcp-adapters: MCP server integrations with automatic tool discovery
- tavily_tools.py: Tavily configuration
- mcp_tools.py: MCP server configs and client wrapper
- tool_registry.py: Central tool loading (combines all tools)

"""

from app.tools.tavily_tools import get_tavily_tools
from app.tools.mcp_tools import get_mcp_client, load_mcp_tools
from app.tools.tool_registry import get_research_tools

__all__ = ["get_tavily_tools", "get_mcp_client", "load_mcp_tools", "get_research_tools"]
