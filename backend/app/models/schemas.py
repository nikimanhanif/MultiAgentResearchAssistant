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


class SourceType(str, Enum):
    """Source type classification for credibility scoring."""
    PEER_REVIEWED = "peer_reviewed"
    PREPRINT = "preprint"
    ACADEMIC = "academic"
    NEWS = "news"
    BLOG = "blog"
    GOVERNMENT = "government"
    ORGANIZATION = "organization"
    BOOK = "book"
    CONFERENCE = "conference"
    THESIS = "thesis"
    WEBSITE = "website"
    OTHER = "other"


class Citation(BaseModel):
    """
    Citation model for sources with credibility scoring.
    """
    # Basic citation information
    source: str = Field(..., description="Source name or identifier")
    url: Optional[str] = Field(None, description="URL to the source")
    title: Optional[str] = Field(None, description="Title of the source")
    authors: Optional[List[str]] = Field(
        None,
        description="List of authors (None if no authors available)"
    )
    year: Optional[int] = Field(None, description="Publication year", ge=1900, le=2100)
    
    # Credibility scoring fields (Phase 7.5)
    credibility_score: Optional[float] = Field(
        None,
        description="Credibility score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    source_type: Optional[SourceType] = Field(
        None,
        description="Classification of source type"
    )
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    publication_date: Optional[datetime] = Field(
        None,
        description="Full publication date"
    )
    venue: Optional[str] = Field(
        None,
        description="Publication venue (journal, conference, etc.)"
    )
    is_peer_reviewed: Optional[bool] = Field(
        None,
        description="Whether the source is peer-reviewed"
    )
    citation_count: Optional[int] = Field(
        None,
        description="Number of citations (if available)",
        ge=0
    )
    credibility_warning: Optional[str] = Field(
        None,
        description="Warning message for low-credibility sources"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source": "Nature",
                "url": "https://nature.com/articles/example",
                "title": "Machine Learning in Healthcare",
                "author": "Smith, J. et al.",
                "year": 2023,
                "credibility_score": 0.95,
                "source_type": "peer_reviewed",
                "doi": "10.1038/example",
                "is_peer_reviewed": True,
                "venue": "Nature Medicine"
            }
        }
    )


class ClarificationQuestions(BaseModel):
    """Model for clarification questions to user."""
    clarification_questions: List[str] = Field(
        ...,
        description="List of 1-3 clarifying questions"
    )
    context: Optional[str] = Field(
        None,
        description="Brief explanation of why these questions are being asked"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "clarification_questions": [
                    "What specific aspect of AI would you like to focus on?",
                    "What time period should the research cover?"
                ],
                "context": "I need to understand the scope and constraints of your research."
            }
        }
    )


class ClarificationResponse(BaseModel):
    """Model for user's clarification answers."""
    answers: Dict[str, str]


class ScopeCompletionCheck(BaseModel):
    """Model for scope completion analysis output.
    
    Used by scope agent to determine if enough information has been gathered.
    """
    is_complete: bool = Field(
        ...,
        description="Whether we have enough information to proceed"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the decision"
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="List of missing information items"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "is_complete": False,
                "reasoning": "Need to clarify time period and depth requirements",
                "missing_info": ["time_period", "research_depth"]
            }
        }
    )


class ReportFormat(str, Enum):
    """Report format types for different use cases.
    
    Created in Phase 1.2, used in Phase 4.2 for report formatting.
    
    Formats map to specific use cases:
    - SUMMARY: General-purpose quick overview
    - COMPARISON: Use Case 2 - Comparative analysis
    - RANKING: Top-N ranked lists with criteria
    - FACT_VALIDATION: Use Case 3 - Fact validation and credibility assessment
    - LITERATURE_REVIEW: Use Case 1 - Structured academic literature review
    - GAP_ANALYSIS: Use Case 4 - Research gap identification
    - OTHER: Fallback format when no specific format is requested or format cannot be determined.
            Uses generic markdown structure with standard sections (Introduction, Findings, Conclusion, References).
    """
    SUMMARY = "summary"  # Simple summary with key findings
    COMPARISON = "comparison"  # Side-by-side comparison with metrics (Use Case 2)
    RANKING = "ranking"  # Ranked items with criteria and justification
    FACT_VALIDATION = "fact_validation"  # Claims with validation results (Use Case 3)
    LITERATURE_REVIEW = "literature_review"  # Structured academic review (Use Case 1)
    GAP_ANALYSIS = "gap_analysis"  # Research gap identification (Use Case 4)
    OTHER = "other"  # Fallback format - generic markdown structure when format is unspecified


class ResearchBrief(BaseModel):
    """Research brief generated by scope agent."""
    scope: str = Field(..., description="Main research scope/question")
    sub_topics: List[str] = Field(
        ...,
        description="List of sub-topics to research"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Research constraints (time period, depth, etc.)"
    )
    deliverables: str = Field(
        ...,
        description="Expected deliverables description"
    )
    format: Optional[ReportFormat] = Field(
        None,
        description="Desired report format"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the research brief"
    )


class Finding(BaseModel):
    """Single research finding with embedded citation.
    
    Replaces SubAgentFindings for Supervisor Loop architecture (Phase 3.7+).
    Each finding represents a single factual claim with its source.
    """
    claim: str = Field(..., description="The factual claim or finding")
    citation: Citation = Field(..., description="Embedded citation with all metadata")
    topic: str = Field(..., description="Sub-topic this finding addresses", min_length=1)
    credibility_score: float = Field(
        ...,
        description="Credibility score inherited from citation (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "claim": "Large language models can be fine-tuned for domain-specific tasks",
                "citation": {
                    "source": "Nature Machine Intelligence",
                    "url": "https://nature.com/articles/s42256-023-00123-4",
                    "title": "Domain Adaptation in Large Language Models",
                    "authors": ["Smith, J.", "Doe, A."],
                    "year": 2023,
                    "credibility_score": 0.95,
                    "source_type": "peer_reviewed"
                },
                "topic": "fine-tuning",
                "credibility_score": 0.95
            }
        }
    )


class ResearchTask(BaseModel):
    """Task assignment for sub-agents in Supervisor Loop task queue."""
    task_id: str = Field(..., description="Unique task identifier", min_length=1)
    topic: str = Field(..., description="Sub-topic to research", min_length=1)
    query: str = Field(..., description="Search query for this task", min_length=1)
    priority: int = Field(..., description="Task priority (1=highest)", ge=1)
    requested_by: Optional[str] = Field(
        None,
        description="Sub-agent ID if this task was delegated by another sub-agent"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task_id": "task_001",
                "topic": "fine-tuning methods",
                "query": "LLM fine-tuning techniques for domain adaptation",
                "priority": 1,
                "requested_by": None
            }
        }
    )


class GapType(str, Enum):
    """Types of research gaps identified."""
    COVERAGE = "coverage"  # Missing sub-topics from brief
    DEPTH = "depth"  # Insufficient sources per topic
    QUALITY = "quality"  # Low average credibility score
    TEMPORAL = "temporal"  # Missing time periods from constraints
    PERSPECTIVE = "perspective"  # Missing viewpoints or approaches


class ResearchGap(BaseModel):
    """Model for identified research gaps.
    
    Used in Phase 8.5 for gap analysis and conditional re-research.
    """
    gap_type: GapType = Field(..., description="Type of gap identified")
    description: str = Field(..., description="Description of the gap")
    severity: float = Field(
        ...,
        description="Severity score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    affected_topics: List[str] = Field(
        default_factory=list,
        description="Sub-topics affected by this gap"
    )
    recommendation: Optional[str] = Field(
        None,
        description="Recommendation to address the gap"
    )


class CoverageAnalysis(BaseModel):
    """Analysis of research coverage.
    
    Used in Phase 8.5 for gap identification and Phase 4.2 for report formatting.
    """
    total_topics: int = Field(..., description="Total number of sub-topics", ge=0)
    covered_topics: int = Field(..., description="Number of covered topics", ge=0)
    coverage_percentage: float = Field(
        ...,
        description="Percentage of topics covered (0.0-100.0)",
        ge=0.0,
        le=100.0
    )
    average_sources_per_topic: float = Field(
        ...,
        description="Average number of sources per topic",
        ge=0.0
    )
    average_credibility: float = Field(
        ...,
        description="Average credibility score across all sources (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    topic_coverage: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of sources per topic"
    )
    temporal_coverage: Optional[Dict[str, Any]] = Field(
        None,
        description="Time period coverage analysis"
    )


class ReviewAction(BaseModel):
    """Model for user review actions on generated reports (Phase 6.2).
    
    Used in HITL reviewer node to capture user feedback and routing decisions.
    """
    action: str = Field(
        ...,
        description="Action to take: approve, refine, or re_research",
        pattern="^(approve|refine|re_research)$"
    )
    feedback: Optional[str] = Field(
        None,
        description="Feedback for refinement or new research query"
    )
    
    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is one of the allowed values."""
        if v not in ["approve", "refine", "re_research"]:
            raise ValueError("Action must be one of: approve, refine, re_research")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "refine",
                "feedback": "Please add more details about recent developments"
            }
        }
    )
