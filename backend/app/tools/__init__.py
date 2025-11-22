"""Tools module for research agent integrations.

This module provides tool integrations using native LangChain packages:
- langchain-tavily: Web search (TavilySearch, TavilyExtract)
- langchain-mcp-adapters: MCP server integrations with automatic tool discovery

Architecture (Phase 2.2 Refactored):
- tavily_tools.py: Tavily configuration
- mcp_tools.py: MCP server configs and client wrapper
- tool_registry.py: Central tool loading (combines all tools)

Implementation: Phase 7
"""

# TODO: Phase 7.4 - Export get_research_tools from tool_registry
# from app.tools.tool_registry import get_research_tools
#
# __all__ = ["get_research_tools"]
