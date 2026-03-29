"""
Deterministic interrupt-handling policies for unattended benchmark runs.

The production graph includes HITL interrupts at scope_wait (clarification)
and reviewer (report review). These policies provide automatic responses
so the graph can run end-to-end without human input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from evals.models import BenchmarkCase, EvalResult, RunMetadata, RunStatus


# ---------------------------------------------------------------------------
# Clarification Policy
# ---------------------------------------------------------------------------

class BaseClarificationPolicy(ABC):
    """Strategy for responding to scope_wait clarification interrupts."""

    @abstractmethod
    def respond(self, case: BenchmarkCase, state: Dict[str, Any]) -> str:
        """Return a string to resume the graph from a clarification interrupt."""
        ...


class DefaultClarificationPolicy(BaseClarificationPolicy):
    """
    Uses benchmark case context/constraints if available; otherwise falls
    back to a deterministic generic response that guides the scope agent
    toward the broadest reasonable interpretation.
    """

    FALLBACK = (
        "Proceed with the broadest reasonable interpretation of the query. "
        "Cover all major sub-topics. No time-period constraints. "
        "Aim for a comprehensive literature-review style report."
    )

    def respond(self, case: BenchmarkCase, state: Dict[str, Any]) -> str:
        parts: list[str] = []
        if case.context:
            parts.append(case.context)
        if case.constraints:
            constraints_str = ", ".join(
                f"{k}: {v}" for k, v in case.constraints.items()
            )
            parts.append(f"Constraints: {constraints_str}")
        if case.expected_topics:
            parts.append(f"Focus on: {', '.join(case.expected_topics)}")
        return " ".join(parts) if parts else self.FALLBACK


# ---------------------------------------------------------------------------
# Review Policy
# ---------------------------------------------------------------------------

class BaseReviewPolicy(ABC):
    """Strategy for responding to reviewer interrupts."""

    @abstractmethod
    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a resume dict (action + optional feedback)."""
        ...


class AutoApproveReviewPolicy(BaseReviewPolicy):
    """Auto-approves the generated report. Default for benchmarks."""

    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "approve", "feedback": None}


# ---------------------------------------------------------------------------
# Failure Policy
# ---------------------------------------------------------------------------

class BaseFailurePolicy(ABC):
    """Strategy for handling partial or failed runs."""

    @abstractmethod
    def handle(
        self,
        case: BenchmarkCase,
        error: Exception,
        partial_state: Optional[Dict[str, Any]],
        run_metadata: RunMetadata,
    ) -> EvalResult:
        """Return an EvalResult representing the failed run."""
        ...


class DefaultFailurePolicy(BaseFailurePolicy):
    """Records the error and returns a FAILURE result with whatever partial data exists."""

    def handle(
        self,
        case: BenchmarkCase,
        error: Exception,
        partial_state: Optional[Dict[str, Any]],
        run_metadata: RunMetadata,
    ) -> EvalResult:
        run_metadata.error_log.append(str(error))
        report = None
        findings_count = 0
        if partial_state:
            report = partial_state.get("report_content")
            findings_count = len(partial_state.get("findings", []))
        return EvalResult(
            case_id=case.case_id,
            status=RunStatus.PARTIAL if report else RunStatus.FAILURE,
            query=case.query,
            report_content=report,
            findings_count=findings_count,
            metadata=run_metadata,
            case_metadata=case.metadata,
        )
