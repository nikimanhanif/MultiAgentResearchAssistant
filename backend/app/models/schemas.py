"""Pydantic schemas for request/response models."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User message", min_length=1)
    conversation_id: Optional[str] = Field(
        None, description="Conversation identifier for chat history"
    )
    deep_research: bool = Field(
        False, description="Enable deep research mode for more thorough analysis"
    )
    enabled_mcp_servers: Optional[List[str]] = Field(
        None, description="List of enabled MCP server names from frontend"
    )

    @field_validator("message", mode="before")
    @classmethod
    def validate_message_not_whitespace(cls, v: str) -> str:
        """Validate that message is not whitespace-only."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Message cannot be empty or whitespace-only")
        return v

    @field_validator("deep_research", mode="before")
    @classmethod
    def validate_deep_research_is_bool(cls, v) -> bool:
        """Validate that deep_research is a boolean."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            raise ValueError("deep_research must be a boolean, not a string")
        if isinstance(v, (int, float)):
            return bool(v)
        raise ValueError("deep_research must be a boolean")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "What is machine learning?",
                "conversation_id": "conv_123",
                "deep_research": False,
            }
        }
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation identifier", min_length=1)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    response_type: Optional[str] = None  # CLARIFICATION, RESEARCH_PROGRESS, REPORT
    scope_status: Optional[str] = None  # CLARIFYING, COMPLETE

    @field_validator("conversation_id", mode="before")
    @classmethod
    def validate_conversation_id_not_whitespace(cls, v: str) -> str:
        """Validate that conversation_id is not whitespace-only."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Conversation ID cannot be empty or whitespace-only")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Machine learning is a subset of artificial intelligence...",
                "conversation_id": "conv_123",
                "timestamp": "2024-01-01T12:00:00",
            }
        }
    )


class ResearchRequest(BaseModel):
    """Request model for research tasks (for future use)."""

    query: str = Field(..., description="Research query", min_length=1)
    context: Optional[str] = Field(None, description="Additional context for research")
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=100)

    @field_validator("query", mode="before")
    @classmethod
    def validate_query_not_whitespace(cls, v: str) -> str:
        """Validate that query is not whitespace-only."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Latest developments in quantum computing",
                "context": "Focus on hardware implementations",
                "max_results": 5,
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response model for API error handling."""

    error: str = Field(..., description="Error message", min_length=1)
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    @field_validator("error", mode="before")
    @classmethod
    def validate_error_not_whitespace(cls, v: str) -> str:
        """Validate that error is not whitespace-only."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Error message cannot be empty or whitespace-only")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Validation error",
                "detail": "Message field is required",
                "timestamp": "2024-01-01T12:00:00",
            }
        }
    )


# New models for agent pipeline (minimal stubs - implementation deferred)

class ScopeStatus(str, Enum):
    """Status of scope clarification phase."""
    CLARIFYING = "CLARIFYING"
    COMPLETE = "COMPLETE"


class Citation(BaseModel):
    """Citation model for sources."""
    source: str
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None


class ClarificationQuestions(BaseModel):
    """Model for clarification questions to user."""
    questions: List[str]
    context: Optional[str] = None


class ClarificationResponse(BaseModel):
    """Model for user's clarification answers."""
    answers: Dict[str, str]


class ResearchBrief(BaseModel):
    """Research brief generated by scope agent."""
    scope: str
    sub_topics: List[str]
    constraints: Dict[str, Any]
    deliverables: str
    format: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SubAgentTask(BaseModel):
    """Task assignment for sub-agents."""
    topic: str
    scope: str
    tools: List[str]
    priority: Optional[int] = None


class SubAgentFindings(BaseModel):
    """Structured output from sub-agents."""
    topic: str
    summary: str
    key_facts: List[str]
    citations: List[Citation]
    sources: List[str]
    raw_data: Optional[Dict[str, Any]] = None


class SummarizedFindings(BaseModel):
    """Final summarized findings from research agent."""
    summary: str
    key_findings: List[str]
    sub_topic_findings: List[SubAgentFindings]
    sources: List[Citation]
    research_metadata: Dict[str, Any]

