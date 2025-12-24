"""
Chat API - Unified endpoint for the research pipeline.

Provides a single streaming endpoint that handles the entire research workflow,
from scope clarification to report review, using Server-Sent Events (SSE).
Supports token-by-token streaming using LangGraph's stream_mode="messages".
"""

import logging
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.types import Command

from app.persistence import save_conversation

from app.graphs.research_graph import build_research_graph
from app.models.schemas import ReviewAction
from app.api.streaming import (
    StreamEventType,
    create_sse_event,
    create_token_event,
    create_progress_event,
    create_error_event,
    create_complete_event,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class MessageItem(BaseModel):
    """Individual message in conversation."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="Current user message")
    messages: Optional[List[MessageItem]] = Field(None, description="Full conversation history")
    thread_id: Optional[str] = Field(None, description="Thread ID for continuing conversation")
    user_id: Optional[str] = Field(None, description="User ID for persistence")


async def stream_graph_with_tokens(
    graph: Any,
    input_data: Any,
    config: Dict[str, Any],
    user_query: str = "",
    user_id: str = "default_user",
    thread_id: str = ""
) -> AsyncGenerator[str, None]:
    """
    Stream graph execution with token-by-token LLM output.
    
    Uses LangGraph's stream_mode="messages" for token streaming combined
    with "updates" for state change events.
    
    Args:
        graph: Compiled research graph.
        input_data: Initial state or Command for resumption.
        config: Configuration with thread_id for checkpointing.
        
    Yields:
        str: SSE-formatted event strings.
    """
    current_phase = "scoping"
    phase_start_time = time.time()
    tasks_count = 0
    findings_count = 0
    
    # Track state for persistence
    research_brief = None
    findings_list = []
    report_content = ""
    is_research_complete = False
    
    try:
        # Stream with multiple modes: messages for tokens, updates for state
        async for chunk in graph.astream(
            input_data, 
            config, 
            stream_mode=["messages", "updates"]
        ):
            # Check the type of chunk
            if isinstance(chunk, tuple) and len(chunk) == 2:
                mode, data = chunk
                
                if mode == "messages":
                    # Token streaming from LLM
                    message, metadata = data
                    if hasattr(message, 'content') and message.content:
                        # Determine which node is streaming
                        node_name = metadata.get('langgraph_node', 'assistant')
                        
                        # Only stream tokens for report_agent
                        if node_name == "report_agent":
                            yield create_sse_event(StreamEventType.REPORT_TOKEN, {
                                "content": message.content
                            })
                
                elif mode == "updates":
                    # State updates from nodes
                    for node_name, node_output in data.items():
                        if not isinstance(node_output, dict):
                            continue
                        
                        # Track phase based on node
                        previous_phase = current_phase
                        if node_name == "scope":
                            current_phase = "scoping"
                            # For scope node, send the formatted message content instead of raw tokens
                            if "messages" in node_output and node_output["messages"]:
                                for msg in node_output["messages"]:
                                    if msg.get("role") == "assistant":
                                        content = msg.get("content", "")
                                        if content:
                                            # Send as complete message, not tokens
                                            yield create_sse_event(StreamEventType.TOKEN, {
                                                "content": content,
                                                "node": "scope"
                                            })
                        elif node_name == "supervisor":
                            current_phase = "researching"
                        elif node_name == "report_agent":
                            current_phase = "generating_report"
                        elif node_name == "reviewer":
                            current_phase = "review"
                        
                        # Reset timer on phase change
                        if current_phase != previous_phase:
                            phase_start_time = time.time()
                        
                        # Track progress metrics
                        if "task_history" in node_output:
                            tasks = node_output.get("task_history", [])
                            if tasks:
                                tasks_count = len(tasks)
                        
                        if "findings" in node_output:
                            findings = node_output.get("findings", [])
                            if findings:
                                findings_count = len(findings)
                        
                        # Calculate phase duration in milliseconds
                        phase_duration_ms = int((time.time() - phase_start_time) * 1000)
                        
                        # Emit progress event
                        iterations = node_output.get("budget", {}).get("iterations", 0)
                        yield create_progress_event(
                            phase=current_phase,
                            tasks_count=tasks_count,
                            findings_count=findings_count,
                            iterations=iterations,
                            phase_duration_ms=phase_duration_ms
                        )
                        
                        # Check for research brief creation
                        if "research_brief" in node_output and node_output["research_brief"]:
                            brief = node_output["research_brief"]
                            research_brief = brief  # Track for persistence
                            yield create_sse_event(StreamEventType.BRIEF_CREATED, {
                                "scope": brief.scope if hasattr(brief, 'scope') else str(brief),
                                "sub_topics": brief.sub_topics if hasattr(brief, 'sub_topics') else []
                            })
                        
                        # Track findings for persistence
                        if "findings" in node_output and node_output["findings"]:
                            findings_list = node_output["findings"]
                        
                        # Check for review request (interrupt)
                        if node_name == "reviewer":
                            report = node_output.get("report_content", "")
                            if report:
                                report_content = report  # Track for persistence
                                yield create_sse_event(StreamEventType.REVIEW_REQUEST, {
                                    "report": report
                                })
                        
                        # Track report content
                        if "report_content" in node_output and node_output["report_content"]:
                            report_content = node_output["report_content"]
                        
                        # Track completion status
                        if node_output.get("is_complete", False):
                            is_research_complete = True
                        
                        # State update event
                        yield create_sse_event(StreamEventType.STATE_UPDATE, {
                            "node": node_name,
                            "is_complete": node_output.get("is_complete", False)
                        })
            else:
                # Fallback for simple stream mode
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        yield create_sse_event(StreamEventType.STATE_UPDATE, {
                            "node": node_name
                        })
        
        # Save completed conversation to persistence
        if is_research_complete and research_brief and report_content:
            try:
                await save_conversation(
                    user_id=user_id,
                    conversation_id=thread_id,
                    user_query=user_query,
                    research_brief=research_brief,
                    findings=findings_list,
                    report_content=report_content
                )
                logger.info(f"Saved conversation {thread_id} for user {user_id}")
            except Exception as save_error:
                logger.error(f"Failed to save conversation: {save_error}")
        
        yield create_complete_event()
        
    except Exception as e:
        logger.error(f"Error streaming graph: {e}", exc_info=True)
        yield create_error_event(str(e))


@router.post("")
async def chat(request: ChatRequest):
    """
    Unified chat endpoint for the research pipeline with token streaming.
    
    Handles the complete workflow:
    1. Scope clarification (if new thread or no brief).
    2. Research execution (supervisor loop).
    3. Report generation.
    4. HITL review.
    
    For continuing conversations (same thread_id), the new message is appended
    to the existing conversation history using the messages reducer pattern.
    
    Returns:
        StreamingResponse: Server-Sent Events (SSE) stream with token-by-token output.
    """
    thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:12]}"
    user_id = request.user_id or "default_user"
    
    
    try:
        graph = build_research_graph()
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize research pipeline: {str(e)}"
        )
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }
    
    # Build messages list - use full history if provided, otherwise just the new message
    if request.messages:
        # Frontend sent full conversation history (like Streamlit approach)
        conversation_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        # Add the new message at the end
        conversation_messages.append({
            "role": "user",
            "content": request.message
        })
        logger.info(f"Thread {thread_id}: Received {len(request.messages)} history messages + 1 new message")
    else:
        # First message - no history
        conversation_messages = [{
            "role": "user",
            "content": request.message
        }]
        logger.info(f"Thread {thread_id}: First message, no history")
    
    # Debug: Log all messages being sent to graph
    logger.info(f"=== CONVERSATION MESSAGES ({len(conversation_messages)} total) ===")
    for i, msg in enumerate(conversation_messages):
        logger.info(f"  [{i}] {msg['role']}: {msg['content'][:100]}...")
    logger.info("=== END MESSAGES ===")
    
    # Build input state with full conversation
    input_data = {
        "messages": conversation_messages,
        "budget": {
            "iterations": 0,
            "max_iterations": 20,
            "max_sub_agents": 20,
            "max_searches_per_agent": 2,
            "total_searches": 0
        }
    }
    
    # Extract original user query for persistence
    original_query = request.message
    
    return StreamingResponse(
        stream_graph_with_tokens(
            graph, 
            input_data, 
            config,
            user_query=original_query,
            user_id=user_id,
            thread_id=thread_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Thread-ID": thread_id,
            "X-User-ID": user_id
        }
    )


@router.post("/{thread_id}/resume")
async def resume_review(thread_id: str, action: ReviewAction):
    """
    Resume a paused run with HITL reviewer feedback.
    
    Uses LangGraph's Command(resume=...) pattern to continue
    execution from an interrupt point.
    
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
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize research pipeline: {str(e)}"
        )
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # Use Command(resume=...) pattern per LangGraph docs
    resume_value = {
        "action": action.action,
        "feedback": action.feedback
    }
    
    return StreamingResponse(
        stream_graph_with_tokens(graph, Command(resume=resume_value), config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Thread-ID": thread_id
        }
    )


@router.get("/user")
async def get_or_create_user():
    """
    Generate a new user ID.
    
    Returns a UUID that can be stored client-side for conversation persistence.
    """
    return {"user_id": f"user_{uuid.uuid4().hex[:12]}"}
