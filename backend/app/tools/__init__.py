"""Tools module for research agent integrations.

This module provides tool integrations using native LangChain packages:
- langchain-mcp-adapters: MCP server integrations with automatic tool discovery
- tavily_tools.py: Tavily configuration
- mcp_tools.py: MCP server configs and client wrapper
- tool_registry.py: Central tool loading (combines all tools)

"""

from app.tools.tavily_tools import get_tavily_tools

# TODO: Phase 7.4 - Export get_research_tools from tool_registry
# from app.tools.tool_registry import get_research_tools

__all__ = ["get_tavily_tools"]
