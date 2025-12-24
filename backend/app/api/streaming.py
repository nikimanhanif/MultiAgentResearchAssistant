"""
SSE Streaming Utilities.

Provides helpers for creating Server-Sent Events (SSE) for the chat API.
Defines event types and formatting functions for consistent streaming responses.
"""

import json
from enum import Enum
from typing import Any, Dict


class StreamEventType(str, Enum):
    """Types of events that can be streamed to the client."""
    TOKEN = "token"
    PROGRESS = "progress"
    STATE_UPDATE = "state_update"
    BRIEF_CREATED = "brief_created"
    REPORT_TOKEN = "report_token"
    CLARIFICATION_REQUEST = "clarification_request"
    REVIEW_REQUEST = "review_request"
    COMPLETE = "complete"
    ERROR = "error"


def create_sse_event(event_type: StreamEventType, data: Dict[str, Any]) -> str:
    """
    Create a properly formatted SSE event string.
    
    Args:
        event_type: The type of event.
        data: Event payload data.
        
    Returns:
        str: Formatted SSE event string.
    """
    payload = {"type": event_type.value, **data}
    return f"data: {json.dumps(payload)}\n\n"


def create_token_event(content: str, node: str = "assistant") -> str:
    """Create an SSE event for a streaming token."""
    return create_sse_event(StreamEventType.TOKEN, {
        "content": content,
        "node": node
    })


def create_progress_event(
    phase: str,
    tasks_count: int = 0,
    findings_count: int = 0,
    iterations: int = 0,
    phase_duration_ms: int = 0
) -> str:
    """Create an SSE event for research progress updates."""
    return create_sse_event(StreamEventType.PROGRESS, {
        "phase": phase,
        "tasks_count": tasks_count,
        "findings_count": findings_count,
        "iterations": iterations,
        "phase_duration_ms": phase_duration_ms
    })


def create_error_event(error: str) -> str:
    """Create an SSE event for errors."""
    return create_sse_event(StreamEventType.ERROR, {"error": error})


def create_complete_event(message: str = "Stream complete") -> str:
    """Create an SSE event for stream completion."""
    return create_sse_event(StreamEventType.COMPLETE, {"message": message})
