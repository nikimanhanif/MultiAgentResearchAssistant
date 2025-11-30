"""LangGraph research workflow - Full pipeline with scope, supervisor, report, and reviewer.

This module implements the complete research graph structure using LangGraph's StateGraph.
The graph coordinates the full pipeline from scope clarification to final report review.

Architecture:
- scope_node: Handles clarification conversation and brief generation
- supervisor_node: Analyzes gaps, generates tasks, checks completion
- sub_agent_node: Executes research tasks in parallel (via Send API)
- report_agent_node: Generates final report from aggregated findings
- reviewer_node: HITL review with approve/refine/re-research options

Flow:
START → scope → (wait or supervisor) → [sub_agent (parallel)] → supervisor → ...
       → report → reviewer → (END or refine/re-research loops)
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
    """Route from scope node based on brief completion.
    
    Args:
        state: Current research state
        
    Returns:
        - "supervisor" if research brief exists
        - "END" if still waiting for user input
    """
    brief = state.get("research_brief")
    if brief:
        logger.info("Scope complete, routing to supervisor")
        return "supervisor"
    else:
        logger.info("Scope incomplete, waiting for user input")
        return END


def route_from_supervisor(state: ResearchState) -> List[Send] | Literal["report_agent", "END"]:
    """Route from supervisor based on completion status.
    
    Args:
        state: Current research state
        
    Returns:
        - "report_agent" if research is complete
        - List of Send objects for parallel sub-agent execution
        - "END" if budget exhausted with no tasks
    """
    # Check completion flag
    if state.get("is_complete", False):
        logger.info("Research complete, routing to report agent")
        return "report_agent"
    
    # Budget enforcement
    budget = state["budget"]
    iterations = budget.get("iterations", 0)
    max_iterations = budget.get("max_iterations", 20)
    
    findings_count = len(state.get("findings", []))
    max_sub_agents = budget.get("max_sub_agents", 20)
    
    if iterations >= max_iterations or findings_count >= max_sub_agents:
        logger.info("Budget exhausted, routing to report")
        return "report_agent"
    
    # Identify pending tasks
    task_history = state.get("task_history", [])
    completed_tasks = set(state.get("completed_tasks", []))
    failed_tasks = set(state.get("failed_tasks", []))
    
    pending_tasks = [
        task for task in task_history
        if task.task_id not in completed_tasks and task.task_id not in failed_tasks
    ]
    
    # If no pending tasks, route to report
    if not pending_tasks:
        logger.info("No pending tasks, routing to report")
        return "report_agent"
    
    # Spawn parallel sub-agents with full state context
    logger.info(f"Routing {len(pending_tasks)} tasks to sub-agents")
    return [
        Send("sub_agent", {
            "task": task,
            "budget": state["budget"],
            "research_brief": state.get("research_brief")
        })
        for task in pending_tasks
    ]


def build_research_graph() -> StateGraph:
    """Build the complete research workflow graph.
    
    Creates a compiled LangGraph with the full pipeline:
    - scope node for clarification and brief generation
    - supervisor node for gap analysis and task generation
    - sub_agent node for parallel research execution
    - report_agent node for report generation
    - reviewer node for HITL feedback
    
    Returns:
        Compiled StateGraph ready for execution with checkpointer
    """
    logger.info("Building research graph")
    
    # Create graph builder
    builder = StateGraph(ResearchState)
    
    # Add all nodes
    builder.add_node("scope", scope_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sub_agent", sub_agent_node)
    builder.add_node("report_agent", report_agent_node)
    builder.add_node("reviewer", reviewer_node)
    
    # Define edges
    # START -> scope (entry point)
    builder.add_edge(START, "scope")
    
    # scope -> supervisor or END (conditional)
    builder.add_conditional_edges(
        "scope",
        route_from_scope,
        {"supervisor": "supervisor", END: END}
    )
    
    # supervisor -> sub_agents or report (conditional)
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["sub_agent", "report_agent"]
    )
    
    # sub_agent -> supervisor (loop back)
    builder.add_edge("sub_agent", "supervisor")
    
    # report_agent -> reviewer
    builder.add_edge("report_agent", "reviewer")
    
    # reviewer handles its own routing via Command
    # (no static edges needed, reviewer uses dynamic routing)
    
    # Compile with checkpointer for persistence
    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    
    logger.info("Research graph compiled successfully")
    return graph
