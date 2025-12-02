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

def merge_budgets(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """Custom reducer for budget field to handle concurrent updates.
    
    This allows parallel sub-agents to update search counts while
    the supervisor manages iteration counters.
    
    Strategy:
    - Supervisor updates: iterations (uses max to avoid overwrites)
    - Sub-agent updates: total_searches (sum the deltas from each agent)
    - Limits (max_*) should not change during execution
    
    Args:
        left: Current budget state from the graph
        right: Budget update from a node (contains full budget dict)
        
    Returns:
        Merged budget dictionary
    """
    # Start with the right (latest) budget as base (includes limits)
    merged = right.copy()
    
    # For iterations: use max to ensure supervisor updates take precedence
    merged["iterations"] = max(
        left.get("iterations", 0), 
        right.get("iterations", 0)
    )
    
    left_searches = left.get("total_searches", 0)
    right_searches = right.get("total_searches", 0)
    searches_delta = max(0, right_searches - left_searches)  # Compute increment
    merged["total_searches"] = left_searches + searches_delta
    
    return merged

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
        budget: Budget tracking (iterations, max_iterations, max_sub_agents, max_searches_per_agent, total_searches)
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
    
    # Budget tracking with reducer for parallel updates (Phase 8)
    budget: Annotated[Dict[str, int], merge_budgets]
    
    # Gap analysis fields (LLM-based, not algorithmic)
    gaps: Optional[Dict[str, Any]]
    
    # Workflow control
    is_complete: bool
    error: Annotated[List[str], operator.add]  # Changed to list with reducer for parallel agents
    
    # Message history with reducer (for conversation context)
    messages: Annotated[List[Dict[str, Any]], operator.add]
    
    # HITL Reviewer fields
    report_content: str
    reviewer_feedback: Optional[str]


class SubAgentState(ResearchState):
    """State schema for individual sub-agent execution (Phase 8.3).
    
    Extends ResearchState with task-specific field for Send API.
    This enables parallel execution of multiple sub-agents with
    different tasks while sharing the same base state context.
    
    Fields:
        task: The specific ResearchTask assigned to this sub-agent
        (inherits all ResearchState fields)
    """
    task: ResearchTask


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
            "max_sub_agents": 20,
            "max_searches_per_agent": 2,
            "total_searches": 0
        },
        gaps=None,
        is_complete=False,
        error=[],  # Changed to empty list
        messages=[],
        report_content="",
        reviewer_feedback=None
    )
