"""
Data models for the evaluation harness.

Defines the input/output schema for benchmark cases, evaluation results,
run metadata, and score records.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Terminal status of a benchmark run."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    TIMEOUT = "timeout"


class BenchmarkCase(BaseModel):
    """
    A single benchmark input case.

    Adapters convert dataset-specific formats into this shape before
    the runner processes them.
    """
    case_id: str | int = Field(..., description="Unique identifier for this case")
    query: str = Field(..., description="Research query to submit to the pipeline")
    context: Optional[str] = Field(
        None,
        description="Additional context for clarification policy fallback"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        None,
        description="Research constraints (time period, depth, domain, etc.)"
    )
    expected_answer: Optional[str] = Field(
        None,
        description="Reference answer for scoring, if available"
    )
    expected_topics: Optional[List[str]] = Field(
        None,
        description="Expected sub-topics the report should cover"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (source dataset, difficulty, field, etc.)"
    )


class ScoreResult(BaseModel):
    """A single named metric produced by a scorer."""
    metric: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    detail: Optional[str] = Field(None, description="Human-readable detail")


class RunMetadata(BaseModel):
    """Operational metadata captured during a benchmark run."""
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    clarification_interrupts: int = 0
    review_interrupts: int = 0
    graph_iterations: int = 0
    finding_count: int = 0
    error_log: List[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """
    Complete evaluation result linking a benchmark case to its output,
    scores, and run metadata.
    """
    case_id: str | int
    status: RunStatus
    query: str
    report_content: Optional[str] = None
    findings_count: int = 0
    scores: List[ScoreResult] = Field(default_factory=list)
    metadata: RunMetadata = Field(default_factory=RunMetadata)
    case_metadata: Dict[str, Any] = Field(default_factory=dict)

    def score_by_name(self, metric: str) -> Optional[float]:
        """Look up a score value by metric name."""
        for s in self.scores:
            if s.metric == metric:
                return s.value
        return None
