"""Sub-Agent - Research execution with delegation capability.

These agents execute specific research tasks assigned by the supervisor:
- Use tools (Tavily, MCP servers) to gather information
- Extract key facts and create Finding objects with embedded citations
- Score citations using LLM with heuristics
- Can request further research via delegation tool

Architecture: Supervisor Loop (Phase 8.3)
- Input: ResearchTask (via LangGraph Send API)
- Output: List[Finding] with embedded citations
- Delegation: Sub-agent can add new tasks to task_history

Future implementation:
- sub_agent_node(state: SubAgentState) -> Dict
- Tool execution: Research topic using selected tools
- Citation scoring: Single LLM call with credibility heuristics
- Finding creation: Extract facts, embed citations, score credibility
- Delegation tool: request_further_research(topic, reason) adds to task_history
- Return type: List[Finding] (not SubAgentFindings)
"""

