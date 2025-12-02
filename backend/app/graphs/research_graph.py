"""
LangGraph research workflow - Full pipeline.

Implements the complete research graph structure using LangGraph's StateGraph.
Coordinates the pipeline from scope clarification to final report review.

Architecture:
- scope_node: Handles clarification conversation and brief generation.
- supervisor_node: Analyzes gaps, generates tasks, checks completion.
- sub_agent_node: Executes research tasks in parallel (via Send API).
- report_agent_node: Generates final report from aggregated findings.
- reviewer_node: HITL review with approve/refine/re-research options.

Flow:
START -> scope -> (wait or supervisor) -> [sub_agent (parallel)] -> supervisor -> ...
       -> report -> reviewer -> (END or refine/re-research loops)
"""

from typing import List, Literal
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.graphs.state import ResearchState, SubAgentState
from app.agents.scope_agent import scope_node
from app.agents.supervisor_agent import supervisor_node
from app.agents.sub_agent import sub_agent_node
from app.agents.report_agent import report_agent_node
from app.agents.reviewer_agent import reviewer_node
from app.persistence.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)


def route_from_scope(state: ResearchState) -> Literal["supervisor", "END"]:
    """
    Route from scope node based on brief completion.
    
    Args:
        state: Current research state.
        
    Returns:
        Literal["supervisor", "END"]: Next node or END if waiting for input.
    """
    if state.get("research_brief"):
        return "supervisor"
    return END


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
    
    budget = state["budget"]
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
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sub_agent", sub_agent_node)
    builder.add_node("report_agent", report_agent_node)
    builder.add_node("reviewer", reviewer_node)
    
    builder.add_edge(START, "scope")
    
    builder.add_conditional_edges(
        "scope",
        route_from_scope,
        {"supervisor": "supervisor", END: END}
    )
    
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["sub_agent", "report_agent"]
    )
    
    builder.add_edge("sub_agent", "supervisor")
    
    builder.add_edge("report_agent", "reviewer")
    
    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph
