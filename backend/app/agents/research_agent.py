"""Research Agent - Supervisor Loop with task queue.

This agent manages the research phase using a Supervisor Loop architecture:
- Supervisor analyzes gaps in findings and generates new research tasks
- Sub-agents execute tasks in parallel using tools (Tavily, MCP servers)
- Sub-agents can request further research via delegation tool
- Loop continues until complete or budget exhausted

Architecture: Supervisor Loop (Phase 8)
- No strategy selection (prompt-driven, not programmatic)
- Task queue managed by supervisor
- LLM-based gap analysis (not algorithmic)
- Budget limits: max_iterations=20, max_sub_agents=30

Future implementation:
- async def conduct_research(research_brief: ResearchBrief, enabled_tools: List[str]) -> List[Finding]
- Supervisor node: gap analysis, task generation, completion check
- Sub-agent nodes: parallel execution, citation scoring, delegation
- Returns: List[Finding] (filtered and sorted by credibility)
"""

