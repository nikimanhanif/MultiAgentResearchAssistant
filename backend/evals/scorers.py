"""
Internal / preliminary scoring layer for evaluation results.

NOTE: These scorers are internal diagnostic metrics only. They are NOT
equivalent to official DRB evaluation. Official DRB scoring uses:
  - RACE (Reference-based Adaptive Criteria-driven Evaluation) for quality
  - FACT (Framework for Factual Abundance and Citation Trustworthiness) for citations

To run official DRB evaluation, use:
  python -m evals.cli evaluate --drb-repo-path /path/to/deep_research_bench

Each scorer implements the Scorer protocol and produces one or more
ScoreResult objects from an EvalResult, the original BenchmarkCase,
and the final graph state values. All metrics are prefixed with
'internal_' in reports where both internal and official scores appear.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from evals.models import BenchmarkCase, EvalResult, RunStatus, ScoreResult


class Scorer(ABC):
    """Protocol for evaluation scorers."""

    @abstractmethod
    def score(
        self,
        result: EvalResult,
        case: BenchmarkCase,
        state_values: Dict[str, Any],
    ) -> List[ScoreResult]:
        ...


class SuccessScorer(Scorer):
    """Binary score: did the run complete successfully?"""

    def score(
        self, result: EvalResult, case: BenchmarkCase, state_values: Dict[str, Any]
    ) -> List[ScoreResult]:
        return [
            ScoreResult(
                metric="success",
                value=1.0 if result.status == RunStatus.SUCCESS else 0.0,
                detail=result.status.value,
            )
        ]


class LatencyScorer(Scorer):
    """Records wall-clock duration in seconds."""

    def score(
        self, result: EvalResult, case: BenchmarkCase, state_values: Dict[str, Any]
    ) -> List[ScoreResult]:
        duration = (
            result.metadata.duration_seconds
            if result.metadata and result.metadata.duration_seconds is not None
            else 0.0
        )
        return [
            ScoreResult(
                metric="latency_seconds",
                value=duration,
            )
        ]


class FindingScorer(Scorer):
    """Counts findings and topic coverage."""

    def score(
        self, result: EvalResult, case: BenchmarkCase, state_values: Dict[str, Any]
    ) -> List[ScoreResult]:
        findings = state_values.get("findings", [])
        finding_count = len(findings)

        topics_covered = set()
        for f in findings:
            topic = getattr(f, "topic", None)
            if topic:
                topics_covered.add(topic.lower())

        expected_topics = case.expected_topics or []
        if expected_topics:
            matched = sum(
                1 for t in expected_topics if t.lower() in topics_covered
            )
            coverage = matched / len(expected_topics)
        else:
            coverage = 1.0 if finding_count > 0 else 0.0

        return [
            ScoreResult(metric="finding_count", value=float(finding_count)),
            ScoreResult(
                metric="topic_coverage",
                value=coverage,
                detail=f"{len(topics_covered)} unique topics found",
            ),
        ]


class CitationScorer(Scorer):
    """Evaluates citation quality from findings."""

    def score(
        self, result: EvalResult, case: BenchmarkCase, state_values: Dict[str, Any]
    ) -> List[ScoreResult]:
        findings = state_values.get("findings", [])
        if not findings:
            return [
                ScoreResult(metric="citation_count", value=0.0),
                ScoreResult(metric="unique_sources", value=0.0),
                ScoreResult(metric="avg_credibility", value=0.0),
                ScoreResult(metric="citation_url_rate", value=0.0),
            ]

        scores_list: List[ScoreResult] = []
        unique_sources: set[str] = set()
        credibility_values: List[float] = []
        has_url_count = 0

        for f in findings:
            citation = getattr(f, "citation", None)
            if citation:
                source = getattr(citation, "source", None)
                if source:
                    unique_sources.add(source)
                if getattr(citation, "url", None):
                    has_url_count += 1
            cred = getattr(f, "credibility_score", None)
            if cred is not None:
                credibility_values.append(cred)

        avg_cred = (
            sum(credibility_values) / len(credibility_values)
            if credibility_values
            else 0.0
        )

        scores_list.append(
            ScoreResult(metric="citation_count", value=float(len(findings)))
        )
        scores_list.append(
            ScoreResult(
                metric="unique_sources",
                value=float(len(unique_sources)),
                detail=", ".join(sorted(unique_sources)[:10]),
            )
        )
        scores_list.append(
            ScoreResult(metric="avg_credibility", value=round(avg_cred, 4))
        )
        scores_list.append(
            ScoreResult(
                metric="citation_url_rate",
                value=round(has_url_count / len(findings), 4) if findings else 0.0,
            )
        )

        return scores_list


class CompletenessScorer(Scorer):
    """
    INTERNAL diagnostic: basic completeness check against a reference answer.

    Uses simple token overlap (Jaccard similarity on whitespace tokens).
    This is NOT equivalent to DRB RACE evaluation. For official quality
    scoring, use the DRB evaluator (python -m evals.cli evaluate).
    """

    def score(
        self, result: EvalResult, case: BenchmarkCase, state_values: Dict[str, Any]
    ) -> List[ScoreResult]:
        if not case.expected_answer or not result.report_content:
            return []

        ref_tokens = set(case.expected_answer.lower().split())
        gen_tokens = set(result.report_content.lower().split())

        if not ref_tokens:
            return []

        intersection = ref_tokens & gen_tokens
        union = ref_tokens | gen_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        return [
            ScoreResult(
                metric="completeness_jaccard",
                value=round(jaccard, 4),
                detail=f"token overlap {len(intersection)}/{len(ref_tokens)} ref tokens",
            )
        ]
