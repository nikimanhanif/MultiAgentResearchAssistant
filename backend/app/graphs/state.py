"""LangGraph state definition for research workflow.

This module defines the state structure for the research agent's LangGraph workflow.
The graph coordinates sub-agents in parallel execution with token optimization.

Implemented in Phase 3.5.3 (State & Graph Infrastructure)
Full integration: Phase 8.6

State Fields:
- research_brief: ResearchBrief (from scope agent)
- strategy: Optional[ResearchStrategy] (FLAT/FLAT_REFINEMENT/HIERARCHICAL - Phase 8.1)
- tasks: List[SubAgentTask] (tasks assigned by supervisor)
- findings: Annotated[List[SubAgentFindings], operator.add] (reducer for parallel updates)
- summarized_findings: Optional[SummarizedFindings] (final aggregated output - Phase 8.4)
- gaps: Optional[Dict[str, Any]] (algorithmic gap detection results - Phase 8.5)
- extraction_budget: Dict[str, int] (token optimization - {"used": 0, "max": 5})
- is_complete: bool (whether research is sufficient)
- error: Optional[str] (error message if any)
- messages: Annotated[List[Dict[str, Any]], operator.add] (message history with reducer)

Key Features:
- Reducer Pattern: findings and messages use operator.add for concurrent updates
- Token Optimization: extraction_budget prevents token explosion from full-text papers
- Gap Analysis: gaps field enables conditional re-research routing
- Hierarchical Support: strategy determines single-level vs. multi-level spawning
- Error Handling: error field tracks failures for recovery
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator

from app.models.schemas import (
    ResearchBrief,
    SubAgentTask,
    SubAgentFindings,
    SummarizedFindings,
)


class ResearchState(TypedDict, total=False):
    """State schema for research workflow graph.
    
    Uses TypedDict with reducer annotations for parallel agent updates.
    The 'total=False' allows optional fields.
    
    Reducers:
    - findings: operator.add - Appends findings from parallel sub-agents
    - messages: operator.add - Appends messages from nodes for conversation history
    
    Fields:
        research_brief: Research scope and objectives from scope agent
        strategy: Research strategy (FLAT/FLAT_REFINEMENT/HIERARCHICAL)
        tasks: List of tasks assigned to sub-agents
        findings: Sub-agent findings (uses reducer for parallel updates)
        summarized_findings: Final aggregated findings after compression
        gaps: Detected research gaps for conditional re-research
        extraction_budget: Token budget tracking for paper extraction
        is_complete: Whether research has been completed
        error: Error message if workflow encounters failures
        messages: Message history for conversation context (uses reducer)
    """
    # Core research fields
    research_brief: ResearchBrief
    
    # Strategy and task management (Phase 8.1, 8.2)
    strategy: Optional[str]  # Will use ResearchStrategy model in Phase 8.1
    tasks: List[SubAgentTask]
    
    # Findings with reducer for parallel updates (Phase 8.3, 8.4)
    findings: Annotated[List[SubAgentFindings], operator.add]
    summarized_findings: Optional[SummarizedFindings]
    
    # Gap analysis fields (Phase 8.5)
    gaps: Optional[Dict[str, Any]]
    
    # Token optimization (Phase 8.3, 8.4)
    extraction_budget: Dict[str, int]
    
    # Workflow control
    is_complete: bool
    error: Optional[str]
    
    # Message history with reducer (for conversation context)
    messages: Annotated[List[Dict[str, Any]], operator.add]


def create_initial_state(research_brief: ResearchBrief) -> ResearchState:
    """Create initial research state from research brief.
    
    Args:
        research_brief: Research brief from scope agent
        
    Returns:
        Initial ResearchState with default values
    """
    return ResearchState(
        research_brief=research_brief,
        strategy=None,
        tasks=[],
        findings=[],
        summarized_findings=None,
        gaps=None,
        extraction_budget={"used": 0, "max": 5},
        is_complete=False,
        error=None,
        messages=[]
    )
