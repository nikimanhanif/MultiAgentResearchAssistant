"""
Tools module for research agent integrations.

This module provides tool integrations using native LangChain packages:
- Tavily: Web search and extraction.
- Academic: ArXiv, Semantic Scholar, Scopus paper search, citation graph, and retrieval.
- Registry: Centralized tool loading.
"""

from app.tools.tavily_tools import get_tavily_tools
from app.tools.academic import (
    get_academic_tools,
    search_arxiv,
    search_semantic_scholar,
    search_scopus,
    get_citation_graph,
    fetch_paper_content,
)
from app.tools.tool_registry import get_research_tools

__all__ = [
    "get_tavily_tools",
    "get_academic_tools",
    "search_arxiv",
    "search_semantic_scholar",
    "search_scopus",
    "get_citation_graph",
    "fetch_paper_content",
    "get_research_tools",
]
