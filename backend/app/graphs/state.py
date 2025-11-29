"""LangGraph state definition for research workflow.

This module defines the state structure for the research agent's LangGraph workflow.
The graph uses the Supervisor Loop pattern with task queue and parallel execution.

Updated in Phase 3.7 for Supervisor Loop architecture.
Full implementation: Phase 8

State Fields (Supervisor Loop):
- research_brief: ResearchBrief (from scope agent)
- findings: Annotated[List[Finding], operator.add] (findings with embedded citations)
- task_history: Annotated[List[ResearchTask], operator.add] (immutable task log)
- completed_tasks: Annotated[List[str], operator.add] (completed task IDs)
- failed_tasks: Annotated[List[str], operator.add] (failed task IDs)
- budget: Dict[str, int] (iterations, max_iterations, max_sub_agents)
- gaps: Optional[Dict[str, Any]] (LLM-based gap analysis output)
- is_complete: bool (whether research is sufficient)
- error: Optional[str] (error message if any)
- messages: Annotated[List[Dict[str, Any]], operator.add] (message history with reducer)

Key Features:
- Reducer Pattern: findings, task_history, completed_tasks, and messages use operator.add
- Task Queue: Supervisor generates tasks, sub-agents execute in parallel
- Budget Limits: max_iterations=20, max_sub_agents=30 (~160K tokens total)
- LLM-Based Gap Analysis: Supervisor uses LLM prompts, not algorithmic detection
- Error Handling: error field tracks failures for recovery
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator

from app.models.schemas import (
    ResearchBrief,
    Finding,
    ResearchTask,
)


class ResearchState(TypedDict, total=False):
    """State schema for research workflow graph (Supervisor Loop).
    
    Uses TypedDict with reducer annotations for parallel agent updates.
    The 'total=False' allows optional fields.
    
    Reducers:
    - findings: operator.add - Appends findings from parallel sub-agents
    - task_history: operator.add - Appends tasks (immutable log)
    - completed_tasks: operator.add - Appends completed task IDs
    - failed_tasks: operator.add - Appends failed task IDs
    - messages: operator.add - Appends messages from nodes
    
    Fields:
        research_brief: Research scope and objectives from scope agent
        findings: Sub-agent findings (uses reducer for parallel updates)
        task_history: Immutable log of all research tasks
        completed_tasks: Track completed task IDs
        failed_tasks: Track failed task IDs
        budget: Budget tracking (iterations, max_iterations, max_sub_agents)
        gaps: LLM-based gap analysis output (not algorithmic)
        is_complete: Whether research has been completed
        error: Error message if workflow encounters failures
        messages: Message history for conversation context (uses reducer)
        report_content: Generated report for review 
        reviewer_feedback: Optional feedback from user for refinement
    """
    # Core research fields
    research_brief: ResearchBrief
    
    # Findings with reducer for parallel updates (Phase 8.3, 8.4)
    findings: Annotated[List[Finding], operator.add]
    
    # Task queue for Supervisor Loop (Phase 8.2, 8.3)
    task_history: Annotated[List[ResearchTask], operator.add]
    completed_tasks: Annotated[List[str], operator.add]
    failed_tasks: Annotated[List[str], operator.add]
    
    # Budget tracking (Phase 8)
    budget: Dict[str, int]
    
    # Gap analysis fields (LLM-based, not algorithmic)
    gaps: Optional[Dict[str, Any]]
    
    # Workflow control
    is_complete: bool
    error: Optional[str]
    
    # Message history with reducer (for conversation context)
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # HITL Reviewer fields
    report_content: str
    reviewer_feedback: Optional[str]


def create_initial_state(research_brief: ResearchBrief) -> ResearchState:
    """Create initial research state from research brief.
    
    Args:
        research_brief: Research brief from scope agent
        
    Returns:
        Initial ResearchState with default values for Supervisor Loop
    """
    return ResearchState(
        research_brief=research_brief,
        findings=[],
        task_history=[],
        completed_tasks=[],
        failed_tasks=[],
        budget={
            "iterations": 0,
            "max_iterations": 20,
            "max_sub_agents": 20
        },
        gaps=None,
        is_complete=False,
        error=None,
        messages=[],
        report_content="",
        reviewer_feedback=None
    )
