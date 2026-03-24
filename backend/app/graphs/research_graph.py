"""
LangGraph research workflow - Full pipeline.

Implements the complete research graph structure using LangGraph's StateGraph.
Coordinates the pipeline from scope clarification to final report review.

Architecture:
- scope_node: Generates clarifying questions and stores in state.
- scope_wait_node: Handles interrupt to wait for user response.
- supervisor_node: Analyzes gaps, generates tasks, checks completion.
- sub_agent_node: Executes research tasks in parallel (via Send API).
- report_agent_node: Generates final report from aggregated findings.
- reviewer_node: HITL review with approve/refine/re-research options.

Flow:
START -> scope -> scope_wait (interrupt) -> scope -> ... -> supervisor
       -> [sub_agent (parallel)] -> supervisor -> ... -> report -> reviewer
       -> (END or refine/re-research loops)
"""

from typing import List, Literal
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.graphs.state import ResearchState, SubAgentState
from app.agents.scope_agent import scope_node, scope_wait_node
from app.agents.supervisor_agent import supervisor_node
from app.agents.sub_agent import sub_agent_node
from app.agents.report_agent import report_agent_node
from app.agents.reviewer_agent import reviewer_node
from app.persistence.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)


def route_from_scope(state: ResearchState) -> Literal["supervisor", "scope_wait"]:
    """
    Route from scope node based on state.
    
    Args:
        state: Current research state.
        
    Returns:
        Literal["supervisor", "scope_wait"]: 
            - "supervisor" if research brief is ready
            - "scope_wait" if pending questions need user response
    """
    if state.get("research_brief"):
        return "supervisor"
    if state.get("pending_clarification_questions"):
        return "scope_wait"
    return "scope_wait"


def route_from_scope_wait(state: ResearchState) -> Literal["supervisor", "scope"]:
    """
    Route from scope_wait node after user responds.
    
    Args:
        state: Current research state.
        
    Returns:
        Literal["supervisor", "scope"]:
            - "supervisor" if research brief is ready
            - "scope" if more clarification rounds needed
    """
    if state.get("research_brief"):
        return "supervisor"
    return "scope"


def route_from_supervisor(state: ResearchState) -> List[Send] | Literal["report_agent", "END"]:
    """
    Route from supervisor based on completion status and budget.
    
    Args:
        state: Current research state.
        
    Returns:
        List[Send] | Literal["report_agent", "END"]: 
            - "report_agent" if research is complete or budget exhausted.
            - List of Send objects for parallel sub-agent execution.
            - "END" if no pending tasks remain.
    """
    if state.get("is_complete", False):
        return "report_agent"
    
    budget = state.get("budget", {})
    iterations = budget.get("iterations", 0)
    max_iterations = budget.get("max_iterations", 20)
    findings_count = len(state.get("findings", []))
    max_sub_agents = budget.get("max_sub_agents", 20)
    
    if iterations >= max_iterations or findings_count >= max_sub_agents:
        return "report_agent"
    
    task_history = state.get("task_history", [])
    completed_tasks = set(state.get("completed_tasks", []))
    failed_tasks = set(state.get("failed_tasks", []))
    
    pending_tasks = [
        task for task in task_history
        if task.task_id not in completed_tasks and task.task_id not in failed_tasks
    ]
    
    if not pending_tasks:
        return "report_agent"
    
    return [
        Send("sub_agent", {
            "task": task,
            "budget": state["budget"],
            "research_brief": state.get("research_brief")
        })
        for task in pending_tasks
    ]


def build_research_graph() -> StateGraph:
    """
    Build the complete research workflow graph.
    
    Returns:
        StateGraph: Compiled graph ready for execution with checkpointer.
    """
    builder = StateGraph(ResearchState)
    
    builder.add_node("scope", scope_node)
    builder.add_node("scope_wait", scope_wait_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sub_agent", sub_agent_node)
    builder.add_node("report_agent", report_agent_node)
    builder.add_node("reviewer", reviewer_node)
    
    # Entry point
    builder.add_edge(START, "scope")
    
    # Scope node routes to scope_wait (for user input) or supervisor (if brief ready)
    builder.add_conditional_edges(
        "scope",
        route_from_scope,
        {"supervisor": "supervisor", "scope_wait": "scope_wait"}
    )
    
    # Scope wait node routes back to scope (for more rounds) or supervisor (if brief ready)
    builder.add_conditional_edges(
        "scope_wait",
        route_from_scope_wait,
        {"supervisor": "supervisor", "scope": "scope"}
    )
    
    # Supervisor routes to sub_agents or report
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["sub_agent", "report_agent"]
    )
    
    # Sub-agents loop back to supervisor
    builder.add_edge("sub_agent", "supervisor")
    
    # Report agent goes to reviewer
    builder.add_edge("report_agent", "reviewer")
    
    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)


