"""
Unit tests for evals.scorers and evals.reporters.
"""

import json
import os
from datetime import datetime, timedelta

import pytest

from evals.models import (
    BenchmarkCase,
    EvalResult,
    RunMetadata,
    RunStatus,
    ScoreResult,
)
from evals.scorers import (
    CitationScorer,
    CompletenessScorer,
    FindingScorer,
    LatencyScorer,
    SuccessScorer,
)
from evals.reporters import JSONLReporter, CSVReporter, MarkdownReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    status=RunStatus.SUCCESS,
    findings_count=3,
    report="# Report",
    duration=10.5,
    scores=None,
) -> EvalResult:
    meta = RunMetadata(
        start_time=datetime(2025, 1, 1, 12, 0, 0),
        end_time=datetime(2025, 1, 1, 12, 0, int(duration)),
        duration_seconds=duration,
        finding_count=findings_count,
    )
    return EvalResult(
        case_id="test_1",
        status=status,
        query="test query",
        report_content=report,
        findings_count=findings_count,
        metadata=meta,
        scores=scores or [],
    )


def _make_case(**overrides) -> BenchmarkCase:
    defaults = {"case_id": "c1", "query": "test"}
    defaults.update(overrides)
    return BenchmarkCase(**defaults)


class FakeFinding:
    def __init__(self, topic, credibility_score=0.8, source="TestSource", url=None):
        self.topic = topic
        self.credibility_score = credibility_score
        self.citation = type("C", (), {"source": source, "url": url})()


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

class TestSuccessScorer:

    def test_success_returns_1(self):
        result = _make_result(status=RunStatus.SUCCESS)
        scores = SuccessScorer().score(result, _make_case(), {})
        assert scores[0].value == 1.0

    def test_failure_returns_0(self):
        result = _make_result(status=RunStatus.FAILURE)
        scores = SuccessScorer().score(result, _make_case(), {})
        assert scores[0].value == 0.0


class TestLatencyScorer:

    def test_records_duration(self):
        result = _make_result(duration=42.5)
        scores = LatencyScorer().score(result, _make_case(), {})
        assert scores[0].metric == "latency_seconds"
        assert scores[0].value == 42.5


class TestFindingScorer:

    def test_counts_findings(self):
        findings = [FakeFinding("A"), FakeFinding("B"), FakeFinding("A")]
        scores = FindingScorer().score(
            _make_result(), _make_case(), {"findings": findings}
        )
        count_score = next(s for s in scores if s.metric == "finding_count")
        assert count_score.value == 3.0

    def test_topic_coverage_with_expected(self):
        findings = [FakeFinding("transformers"), FakeFinding("rag")]
        case = _make_case(expected_topics=["transformers", "rag", "summarization"])
        scores = FindingScorer().score(_make_result(), case, {"findings": findings})
        coverage = next(s for s in scores if s.metric == "topic_coverage")
        assert coverage.value == pytest.approx(2 / 3)

    def test_no_findings(self):
        scores = FindingScorer().score(_make_result(), _make_case(), {"findings": []})
        count = next(s for s in scores if s.metric == "finding_count")
        assert count.value == 0.0


class TestCitationScorer:

    def test_with_findings(self):
        findings = [
            FakeFinding("A", 0.9, "ArXiv", "http://example.com"),
            FakeFinding("B", 0.7, "Nature"),
        ]
        scores = CitationScorer().score(
            _make_result(), _make_case(), {"findings": findings}
        )
        metric_map = {s.metric: s.value for s in scores}
        assert metric_map["citation_count"] == 2.0
        assert metric_map["unique_sources"] == 2.0
        assert metric_map["avg_credibility"] == pytest.approx(0.8)
        assert metric_map["citation_url_rate"] == pytest.approx(0.5)

    def test_empty_findings(self):
        scores = CitationScorer().score(
            _make_result(), _make_case(), {"findings": []}
        )
        metric_map = {s.metric: s.value for s in scores}
        assert metric_map["citation_count"] == 0.0


class TestCompletenessScorer:

    def test_with_overlap(self):
        case = _make_case(expected_answer="quantum computing error correction codes")
        result = _make_result(report="quantum computing and error correction methods")
        scores = CompletenessScorer().score(result, case, {})
        assert len(scores) == 1
        assert scores[0].metric == "completeness_jaccard"
        assert 0.0 < scores[0].value < 1.0

    def test_no_reference_returns_empty(self):
        case = _make_case()
        result = _make_result()
        scores = CompletenessScorer().score(result, case, {})
        assert scores == []


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------

class TestJSONLReporter:

    def test_writes_jsonl_file(self, tmp_path):
        results = [_make_result(), _make_result(status=RunStatus.FAILURE)]
        path = JSONLReporter().write(results, str(tmp_path))

        assert os.path.exists(path)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["case_id"] == "test_1"


class TestCSVReporter:

    def test_writes_csv_with_scores(self, tmp_path):
        results = [
            _make_result(scores=[ScoreResult(metric="success", value=1.0)]),
        ]
        path = CSVReporter().write(results, str(tmp_path))

        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "case_id" in content
        assert "success" in content


class TestMarkdownReporter:

    def test_writes_markdown(self, tmp_path):
        results = [
            _make_result(
                scores=[
                    ScoreResult(metric="success", value=1.0),
                    ScoreResult(metric="latency_seconds", value=10.5),
                ]
            ),
        ]
        path = MarkdownReporter().write(results, str(tmp_path))

        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "Evaluation Report" in content
        assert "test_1" in content
        assert "success" in content


class TestDRBOutputReporter:

    def test_writes_drb_format(self, tmp_path):
        from evals.reporters import DRBOutputReporter
        
        results = [
            _make_result(
                report="Extensive research on topic...",
            ),
        ]
        results[0].case_metadata = {"drb_prompt": "Original prompt text"}
        
        path = DRBOutputReporter(model_name="test_model").write(results, str(tmp_path))

        assert os.path.exists(path)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["id"] == "test_1"
        assert parsed["prompt"] == "Original prompt text"
        assert parsed["article"] == "Extensive research on topic..."

    def test_write_append_merge(self, tmp_path):
        out_dir = str(tmp_path)
        from evals.reporters import DRBOutputReporter
        
        reporter = DRBOutputReporter(model_name="merged")
        
        # 1. First run, entries 2 and 4
        res1 = []
        for i in [2, 4]:
            r = _make_result(report=f"Report {i}")
            r.case_id = str(i)
            r.case_metadata = {"drb_prompt": f"Prompt {i}"}
            res1.append(r)
            
        path = reporter.write(res1, out_dir, append=True)
        
        # 2. Second run, entries 1 and 2 (2 is an update)
        res2 = []
        for i in [1, 2]:
            r = _make_result(report=f"Report {i} updated")
            r.case_id = str(i)
            r.case_metadata = {"drb_prompt": f"Prompt {i} updated"}
            res2.append(r)
            
        reporter.write(res2, out_dir, append=True)
        
        # Verify
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
            
        assert len(lines) == 3
        parsed = [json.loads(l) for l in lines]
        
        # Should be sorted numerically by default: 1, 2, 4
        assert parsed[0]["id"] == "1"
        assert parsed[0]["article"] == "Report 1 updated"
        
        assert parsed[1]["id"] == "2"
        assert parsed[1]["article"] == "Report 2 updated"
        
        assert parsed[2]["id"] == "4"
        assert parsed[2]["article"] == "Report 4"
