"""Chat API - Unified endpoint for research pipeline.

This module provides a single streaming endpoint that handles the entire
research workflow from scope clarification to report review.
"""

import logging
import uuid
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.graphs.research_graph import build_research_graph
from app.graphs.state import ResearchState
from app.models.schemas import ReviewAction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message (query or response)")
    thread_id: Optional[str] = Field(None, description="Thread ID for continuing conversation")


class ChatStreamEvent(BaseModel):
    """Streaming event model."""
    event_type: str = Field(..., description="Type of event: state_update, message, error, complete")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")


async def stream_graph_updates(
    graph: Any,
    initial_state: Dict[str, Any],
    config: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """Stream graph state updates to client.
    
    Args:
        graph: Compiled research graph
        initial_state: Initial state for graph execution
        config: Configuration with thread_id for checkpointing
        
    Yields:
        JSON-encoded ChatStreamEvent objects
    """
    try:
        # Stream state updates from graph
        async for state in graph.astream(initial_state, config, stream_mode="updates"):
            # Each state is a dict with node name as key
            for node_name, node_output in state.items():
                # Extract relevant data for streaming
                event_data = {
                    "node": node_name,
                }
                
                # Add messages if present
                if "messages" in node_output:
                    messages = node_output["messages"]
                    if messages:
                        latest_message = messages[-1]
                        event_data["message"] = latest_message.get("content", "")
                        event_data["role"] = latest_message.get("role", "assistant")
                
                # Add brief if completed
                if "research_brief" in node_output and node_output["research_brief"]:
                    event_data["brief_created"] = True
                
                # Add report if generated
                if "report_content" in node_output and node_output["report_content"]:
                    event_data["report"] = node_output["report_content"]
                
                # Check for completion
                if node_output.get("is_complete"):
                    event_data["research_complete"] = True
                
                # Stream the update
                event = ChatStreamEvent(
                    event_type="state_update",
                    data=event_data
                )
                yield f"data: {event.model_dump_json()}\n\n"
        
        # Send completion event
        complete_event = ChatStreamEvent(
            event_type="complete",
            data={"message": "Graph execution complete"}
        )
        yield f"data: {complete_event.model_dump_json()}\n\n"
        
    except Exception as e:
        logger.error(f"Error streaming graph updates: {e}")
        error_event = ChatStreamEvent(
            event_type="error",
            data={"error": str(e)}
        )
        yield f"data: {error_event.model_dump_json()}\n\n"


@router.post("")
async def chat(request: ChatRequest):
    """Unified chat endpoint for research pipeline with streaming.
    
    Handles the complete workflow:
    1. Scope clarification (if new thread or no brief)
    2. Research execution (supervisor loop)
    3. Report generation
    4. HITL review
    
    Returns:
        StreamingResponse with Server-Sent Events
    """
    # Generate or use existing thread_id
    thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"Chat request: thread_id={thread_id}, message_preview={request.message[:50]}")
    
    # Build graph
    try:
        graph = build_research_graph()
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize research pipeline: {str(e)}")
    
    # Create config for checkpointing
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # Prepare initial state with user message
    initial_state = {
        "messages": [{
            "role": "user",
            "content": request.message
        }]
    }
    
    # Stream updates
    return StreamingResponse(
        stream_graph_updates(graph, initial_state, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Thread-ID": thread_id
        }
    )


@router.post("/{thread_id}/resume")
async def resume_review(thread_id: str, action: ReviewAction):
    """Resume a paused run with HITL reviewer feedback.
    
    This endpoint handles reviewer actions (approve/refine/re-research)
    and resumes the graph execution.
    
    Args:
        thread_id: Thread ID of the conversation
        action: Review action (approve, refine, re_research) with optional feedback
        
    Returns:
        StreamingResponse with continued execution
    """
    logger.info(f"Resume review: thread_id={thread_id}, action={action.action}")
    
    # Build graph
    try:
        graph = build_research_graph()
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize research pipeline: {str(e)}")
    
    # Create config
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # Resume with action
    # The graph should be paused at the reviewer node
    # We pass the action as the resume value
    resume_state = {
        "action": action.action,
        "feedback": action.feedback
    }
    
    try:
        # Stream updates from resumed execution
        return StreamingResponse(
            stream_graph_updates(graph, resume_state, config),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Thread-ID": thread_id
            }
        )
    except Exception as e:
        logger.error(f"Failed to resume review: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume review: {str(e)}")
