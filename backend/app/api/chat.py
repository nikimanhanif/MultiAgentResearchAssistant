"""
Chat API - Unified endpoint for the research pipeline.

Provides a single streaming endpoint that handles the entire research workflow,
from scope clarification to report review, using Server-Sent Events (SSE).
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
    """
    Stream graph state updates to the client.
    
    Args:
        graph: Compiled research graph.
        initial_state: Initial state for graph execution.
        config: Configuration with thread_id for checkpointing.
        
    Yields:
        str: JSON-encoded ChatStreamEvent objects as SSE data.
    """
    try:
        async for state in graph.astream(initial_state, config, stream_mode="updates"):
            for node_name, node_output in state.items():
                event_data = {
                    "node": node_name,
                }
                
                if "messages" in node_output:
                    messages = node_output["messages"]
                    if messages:
                        latest_message = messages[-1]
                        event_data["message"] = latest_message.get("content", "")
                        event_data["role"] = latest_message.get("role", "assistant")
                
                if "research_brief" in node_output and node_output["research_brief"]:
                    event_data["brief_created"] = True
                
                if "report_content" in node_output and node_output["report_content"]:
                    event_data["report"] = node_output["report_content"]
                
                if node_output.get("is_complete"):
                    event_data["research_complete"] = True
                
                event = ChatStreamEvent(
                    event_type="state_update",
                    data=event_data
                )
                yield f"data: {event.model_dump_json()}\n\n"
        
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
    """
    Unified chat endpoint for the research pipeline with streaming.
    
    Handles the complete workflow:
    1. Scope clarification (if new thread or no brief).
    2. Research execution (supervisor loop).
    3. Report generation.
    4. HITL review.
    
    Returns:
        StreamingResponse: Server-Sent Events (SSE) stream.
    """
    thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"
    
    try:
        graph = build_research_graph()
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize research pipeline: {str(e)}")
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    initial_state = {
        "messages": [{
            "role": "user",
            "content": request.message
        }]
    }
    
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
    """
    Resume a paused run with HITL reviewer feedback.
    
    Handles reviewer actions (approve/refine/re-research) and resumes
    the graph execution from the interrupted state.
    
    Args:
        thread_id: Thread ID of the conversation.
        action: Review action with optional feedback.
        
    Returns:
        StreamingResponse: Continued execution stream.
    """
    try:
        graph = build_research_graph()
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize research pipeline: {str(e)}")
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    resume_state = {
        "action": action.action,
        "feedback": action.feedback
    }
    
    try:
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
