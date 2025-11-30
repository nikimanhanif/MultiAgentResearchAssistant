"""Human-in-the-loop reviewer node for report review

This module implements the reviewer node using LangGraph's interrupt mechanism
to pause execution and allow user feedback on generated reports.

The reviewer supports three actions:
- approve: Accept the report and end workflow
- refine: Loop back to report agent with refinement feedback
- re_research: Loop back to supervisor for new research tasks
"""

from typing import Literal
from langgraph.types import interrupt, Command
from langgraph.graph import END

from app.graphs.state import ResearchState
from app.models.schemas import ResearchTask


def reviewer_node(state: ResearchState) -> Command[Literal["__end__", "report_agent", "supervisor"]]:
    """Node that pauses execution to let the user review the report.
    
    Uses LangGraph's interrupt mechanism to pause the graph and wait for
    user input. The user can approve, request refinement, or request
    additional research.
    
    Args:
        state: Current research state containing the generated report
        
    Returns:
        Command object with routing decision and state updates
        
    Workflow:
        1. Interrupt execution and send report to user for review
        2. Wait for user input (action: approve/refine/re_research + optional feedback)
        3. Route based on action:
           - approve: END (complete workflow)
           - refine: report_agent (regenerate with feedback)
           - re_research: supervisor (create new research tasks)
    """
    user_input = interrupt(value={
        "type": "review_request",
        "report": state["report_content"]
    })
    
    action = user_input.get("action")
    feedback = user_input.get("feedback")
    
    if action == "approve":
        return Command(goto=END)
    
    elif action == "refine":
        return Command(
            goto="report_agent",
            update={"reviewer_feedback": feedback}
        )
    
    elif action == "re_research":
        new_task = create_new_task(feedback)
        return Command(
            goto="supervisor",
            update={
                "task_history": [new_task],
                "reviewer_feedback": None,
                "is_complete": False
            }
        )
    
    return Command(goto=END)


def create_new_task(feedback: str) -> ResearchTask:
    """Create a new research task from user feedback.
    
    Args:
        feedback: User feedback describing what additional research is needed
        
    Returns:
        ResearchTask object for the supervisor to process
    """
    import uuid
    
    return ResearchTask(
        task_id=f"task_reviewer_{uuid.uuid4().hex[:8]}",
        topic="user_requested",
        query=feedback,
        priority=1,
        requested_by="reviewer"
    )
