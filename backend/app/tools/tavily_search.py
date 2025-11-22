"""Tavily web search tool wrapper.

This tool provides web search capabilities via the Tavily API.
Used by research agent and sub-agents for gathering information.

Future implementation:
- async def search_tavily(query: str, max_results: int = 5) -> List[SearchResult]
- Wrap Tavily API as LangChain tool
- Return structured results with citations
- Error handling for API failures
"""

