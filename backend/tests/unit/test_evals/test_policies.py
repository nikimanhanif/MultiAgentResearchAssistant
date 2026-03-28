"""
Unit tests for evals.policies — deterministic interrupt handling.
"""

import pytest

from evals.models import BenchmarkCase, RunMetadata, EvalResult, RunStatus
from evals.policies import (
    DefaultClarificationPolicy,
    AutoApproveReviewPolicy,
    DefaultFailurePolicy,
)


# ---------------------------------------------------------------------------
# ClarificationPolicy
# ---------------------------------------------------------------------------

class TestDefaultClarificationPolicy:

    def setup_method(self):
        self.policy = DefaultClarificationPolicy()

    def test_uses_context_when_available(self):
        case = BenchmarkCase(
            case_id="c1",
            query="test query",
            context="Focus on NLP applications in healthcare",
        )
        result = self.policy.respond(case, {})
        assert "NLP applications in healthcare" in result

    def test_uses_constraints_when_available(self):
        case = BenchmarkCase(
            case_id="c2",
            query="test query",
            constraints={"time_period": "2020-2025", "depth": "comprehensive"},
        )
        result = self.policy.respond(case, {})
        assert "time_period" in result
        assert "2020-2025" in result

    def test_uses_expected_topics_when_available(self):
        case = BenchmarkCase(
            case_id="c3",
            query="test query",
            expected_topics=["transformers", "RAG"],
        )
        result = self.policy.respond(case, {})
        assert "transformers" in result
        assert "RAG" in result

    def test_combines_all_available_fields(self):
        case = BenchmarkCase(
            case_id="c4",
            query="test",
            context="Some context",
            constraints={"depth": "deep"},
            expected_topics=["topic_a"],
        )
        result = self.policy.respond(case, {})
        assert "Some context" in result
        assert "depth" in result
        assert "topic_a" in result

    def test_fallback_when_nothing_available(self):
        case = BenchmarkCase(case_id="c5", query="bare query")
        result = self.policy.respond(case, {})
        assert result == DefaultClarificationPolicy.FALLBACK
        assert "broadest reasonable interpretation" in result

    def test_state_parameter_is_accepted(self):
        """Policy receives graph state but currently ignores it — ensure no crash."""
        case = BenchmarkCase(case_id="c6", query="q")
        state = {"research_brief": None, "findings": []}
        result = self.policy.respond(case, state)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ReviewPolicy
# ---------------------------------------------------------------------------

class TestAutoApproveReviewPolicy:

    def test_returns_approve_action(self):
        policy = AutoApproveReviewPolicy()
        result = policy.respond({"report_content": "some report"})
        assert result == {"action": "approve", "feedback": None}

    def test_returns_dict_type(self):
        policy = AutoApproveReviewPolicy()
        result = policy.respond({})
        assert isinstance(result, dict)
        assert "action" in result


# ---------------------------------------------------------------------------
# FailurePolicy
# ---------------------------------------------------------------------------

class TestDefaultFailurePolicy:

    def test_records_error_in_metadata(self):
        policy = DefaultFailurePolicy()
        case = BenchmarkCase(case_id="f1", query="failing query")
        error = ValueError("something broke")
        meta = RunMetadata()

        result = policy.handle(case, error, None, meta)

        assert result.status == RunStatus.FAILURE
        assert "something broke" in result.metadata.error_log[0]

    def test_partial_status_when_report_exists(self):
        policy = DefaultFailurePolicy()
        case = BenchmarkCase(case_id="f2", query="partial")
        error = RuntimeError("partial fail")
        meta = RunMetadata()
        partial_state = {"report_content": "some partial report", "findings": []}

        result = policy.handle(case, error, partial_state, meta)

        assert result.status == RunStatus.PARTIAL
        assert result.report_content == "some partial report"

    def test_failure_status_when_no_report(self):
        policy = DefaultFailurePolicy()
        case = BenchmarkCase(case_id="f3", query="total fail")
        error = Exception("crash")
        meta = RunMetadata()
        partial_state = {"findings": []}

        result = policy.handle(case, error, partial_state, meta)

        assert result.status == RunStatus.FAILURE
        assert result.report_content is None

    def test_preserves_case_id_and_query(self):
        policy = DefaultFailurePolicy()
        case = BenchmarkCase(case_id="f4", query="my query")
        result = policy.handle(case, Exception("e"), None, RunMetadata())

        assert result.case_id == "f4"
        assert result.query == "my query"
