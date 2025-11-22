"""LangGraph state definition for research workflow.

This module defines the state structure for the research agent's LangGraph workflow.
The graph coordinates sub-agents in parallel execution with token optimization.

Implementation: Phase 8.6

State Fields (Phase 8.6 Architecture):
- research_brief: ResearchBrief (from scope agent)
- strategy: ResearchStrategy (FLAT/FLAT_REFINEMENT/HIERARCHICAL - Phase 8.1)
- tasks: List[SubAgentTask] (tasks assigned by supervisor)
- findings: Annotated[List[SubAgentFindings], operator.add] (reducer for parallel updates)
- summarized_findings: Optional[SummarizedFindings] (final aggregated output - Phase 8.4)
- gaps: Optional[Dict[str, Any]] (algorithmic gap detection results - Phase 8.5)
- extraction_budget: Dict[str, int] (token optimization - {"used": 0, "max": 5})
- is_complete: bool (whether research is sufficient)

Key Features:
- Reducer Pattern: findings uses operator.add for concurrent sub-agent updates
- Token Optimization: extraction_budget prevents token explosion from full-text papers
- Gap Analysis: gaps field enables conditional re-research routing
- Hierarchical Support: strategy determines single-level vs. multi-level spawning
"""

# TODO: Phase 8.6 - Implement ResearchState TypedDict with all fields above
# TODO: Phase 8.6 - Use Annotated[List, operator.add] for findings reducer
# TODO: Phase 8.6 - Add extraction_budget tracking for paper extraction limits
# TODO: Phase 8.6 - Integrate with PostgresSaver for checkpointing

