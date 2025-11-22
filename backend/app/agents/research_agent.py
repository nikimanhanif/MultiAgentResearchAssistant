"""Research Agent - Coordinates research phase with sub-agents and tools.

This agent receives a research brief, decides how to conduct research,
coordinates sub-agents, uses tools (Tavily, MCP servers), and summarizes findings.

Future implementation:
- async def conduct_research(research_brief: ResearchBrief, enabled_tools: List[str]) -> SummarizedFindings
- Research strategy decision (breakdown, tool selection, sub-agent delegation)
- Sub-agent coordination (via LangGraph or functional coordination)
- Tool usage integration (Tavily, MCP servers)
- Progress monitoring and completion detection
- Findings summarization
- Returns summarized findings (not raw data)
"""

