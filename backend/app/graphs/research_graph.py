"""LangGraph research workflow - Supervisor Loop with parallel sub-agents.

This module implements the research graph structure using LangGraph's StateGraph.
The graph coordinates the supervisor loop pattern with parallel sub-agent execution.

Architecture:
- supervisor_node: Analyzes gaps, generates tasks, checks completion
- sub_agent_node: Executes research tasks in parallel (via Send API)
- route_tasks: Conditional edge function for dynamic routing

Flow:
START → supervisor → route_tasks → [sub_agent (parallel)] → supervisor → ...
                     ↓
                    END (if complete)
"""

from typing import List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.graphs.state import ResearchState, SubAgentState
from app.agents.supervisor_agent import supervisor_node
from app.agents.sub_agent import sub_agent_node


def route_tasks(state: ResearchState) -> List[Send] | Literal["END"]:
    """Conditional edge function to route tasks to sub-agents or end.
    
    This function implements the core routing logic for the supervisor loop:
    1. Check if research is complete → route to END
    2. Check budget limits → route to END if exhausted
    3. Identify pending tasks (in task_history but not completed)
    4. Use Send API to spawn parallel sub-agents for each pending task
    
    Args:
        state: Current research state
        
    Returns:
        - END string if complete or budget exhausted
        - List of Send objects for parallel sub-agent execution
    """
    # Check completion flag
    if state.get("is_complete", False):
        return END
    
    # Budget enforcement: double-check before spawning sub-agents
    budget = state["budget"]
    iterations = budget.get("iterations", 0)
    max_iterations = budget.get("max_iterations", 20)
    
    findings_count = len(state.get("findings", []))
    max_sub_agents = budget.get("max_sub_agents", 20)
    
    if iterations >= max_iterations or findings_count >= max_sub_agents:
        return END
    
    # Identify pending tasks
    task_history = state.get("task_history", [])
    completed_tasks = set(state.get("completed_tasks", []))
    failed_tasks = set(state.get("failed_tasks", []))
    
    pending_tasks = [
        task for task in task_history
        if task.task_id not in completed_tasks and task.task_id not in failed_tasks
    ]
    
    # If no pending tasks, end the loop
    if not pending_tasks:
        return END
    
    # Map each pending task to a sub-agent node via Send
    # Send injects the 'task' field into SubAgentState while preserving context
    return [
        Send("sub_agent", {"task": task})
        for task in pending_tasks
    ]


def build_research_graph() -> StateGraph:
    """Build the research workflow graph with supervisor loop.
    
    Creates a compiled LangGraph with:
    - supervisor node for gap analysis and task generation
    - sub_agent node for parallel research execution
    - route_tasks conditional edge for dynamic routing
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph builder
    builder = StateGraph(ResearchState)
    
    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("sub_agent", sub_agent_node)
    
    # Add edges
    # Start with supervisor
    builder.add_edge(START, "supervisor")
    
    # Conditional routing after supervisor
    # - If complete → END
    # - If tasks pending → spawn parallel sub-agents
    builder.add_conditional_edges(
        "supervisor",
        route_tasks,
        ["sub_agent", END]
    )
    
    # Sub-agents loop back to supervisor for next iteration
    builder.add_edge("sub_agent", "supervisor")
    
    # Compile and return
    graph = builder.compile()
    return graph
