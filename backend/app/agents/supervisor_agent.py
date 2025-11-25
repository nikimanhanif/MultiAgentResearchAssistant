"""Supervisor Agent - LLM-based gap analysis and task generation.

This agent coordinates the research phase in the Supervisor Loop:
- Analyzes gaps in current findings vs research brief (LLM-driven)
- Generates new research tasks for missing topics
- Checks completion criteria or budget limits
- Filters and aggregates final findings

Architecture: Supervisor Loop (Phase 8.2, 8.5)
- No hierarchical spawning (flat task queue)
- No strategy selection (prompt-driven)
- LLM-based gap analysis (not programmatic set-difference)

Future implementation:
- supervisor_node(state: ResearchState) -> Dict
- Gap analysis: LLM analyzes findings vs brief
- Task generation: LLM creates tasks for gaps
- Completion check: LLM or budget limits
- Findings aggregation: LLM filtering + deduplication
"""

