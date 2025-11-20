"""LangGraph state definition for research workflow.

This module defines the state structure for the research agent's LangGraph workflow.
The graph coordinates sub-agents in parallel execution.

Future implementation:
- ResearchState TypedDict with fields:
  - research_brief: ResearchBrief (from scope agent)
  - research_strategy: Optional[Dict] (how research will be conducted)
  - sub_agent_tasks: List[SubAgentTask] (tasks assigned by research agent)
  - findings: List[SubAgentFindings] (collected findings from sub-agents)
  - summarized_findings: Optional[SummarizedFindings] (final summarized output)
  - is_complete: bool (whether research is sufficient)
  - current_step: str (current graph step)
  - metadata: Dict[str, Any] (additional state)
"""

