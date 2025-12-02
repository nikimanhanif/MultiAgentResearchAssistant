"""
LangGraph state definition for research workflow.

Defines the state structure for the research agent's LangGraph workflow,
utilizing the Supervisor Loop pattern with task queue and parallel execution.

Key Features:
- Reducer Pattern: findings, task_history, completed_tasks, and messages use operator.add.
- Task Queue: Supervisor generates tasks, sub-agents execute in parallel.
- Budget Limits: Enforces iteration and search limits.
- LLM-Based Gap Analysis: Supervisor uses LLM prompts for decision making.
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator
from app.models.schemas import (
    ResearchBrief,
    Finding,
    ResearchTask,
)


def merge_budgets(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """
    Custom reducer for budget field to handle concurrent updates.
    
    Handles parallel sub-agent search count updates while preserving
    supervisor iteration counter precedence.
    
    Args:
        left: Current budget state from the graph.
        right: Budget update from a node.
        
    Returns:
        Dict[str, int]: Merged budget dictionary.
    """
    merged = right.copy()
    
    merged["iterations"] = max(
        left.get("iterations", 0), 
        right.get("iterations", 0)
    )
    
    left_searches = left.get("total_searches", 0)
    right_searches = right.get("total_searches", 0)
    searches_delta = max(0, right_searches - left_searches)
    merged["total_searches"] = left_searches + searches_delta
    
    return merged


class ResearchState(TypedDict, total=False):
    """
    State schema for research workflow graph (Supervisor Loop).
    
    Fields using reducers support parallel updates from multiple nodes.
    """
    research_brief: ResearchBrief
    findings: Annotated[List[Finding], operator.add]
    task_history: Annotated[List[ResearchTask], operator.add]
    completed_tasks: Annotated[List[str], operator.add]
    failed_tasks: Annotated[List[str], operator.add]
    budget: Annotated[Dict[str, int], merge_budgets]
    gaps: Optional[Dict[str, Any]]
    is_complete: bool
    error: Annotated[List[str], operator.add]
    messages: Annotated[List[Dict[str, Any]], operator.add]
    report_content: str
    reviewer_feedback: Optional[str]


class SubAgentState(ResearchState):
    """State schema for individual sub-agent execution with task assignment."""
    task: ResearchTask


def create_initial_state(research_brief: ResearchBrief) -> ResearchState:
    """
    Create initial research state from research brief.
    
    Args:
        research_brief: Research brief from scope agent.
        
    Returns:
        ResearchState: Initial state with default values.
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
        error=[],
        messages=[],
        report_content="",
        reviewer_feedback=None
    )
