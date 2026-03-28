"""
Unit tests for DRB-specific functionality:
- Official DRB adapter schema loading
- DRB output export format
- DRB evaluator CLI construction and validation
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from evals.adapters.deep_research_bench import DeepResearchBenchAdapter
from evals.drb_evaluator import (
    DRBEvalConfig,
    _validate_drb_repo,
    _validate_env_vars,
    resolve_drb_repo_path,
)
from evals.models import BenchmarkCase, EvalResult, RunMetadata, RunStatus
from evals.reporters import DRBOutputReporter


# ---------------------------------------------------------------------------
# DRB Adapter — Official Schema
# ---------------------------------------------------------------------------

class TestDRBOfficialSchema:

    def test_loads_official_format(self, tmp_path):
        """Official DRB query.jsonl has {id, topic, language, prompt}."""
        records = [
            {"id": "drb_001", "topic": "NLP", "language": "en", "prompt": "What are recent advances in NLP?"},
            {"id": "drb_002", "topic": "Physics", "language": "zh", "prompt": "量子纠错的最新进展是什么？"},
        ]
        path = tmp_path / "query.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

        adapter = DeepResearchBenchAdapter()
        cases = adapter.load_cases(str(path))

        assert len(cases) == 2

        c1 = cases[0]
        assert c1.case_id == "drb_001"
        assert c1.query == "What are recent advances in NLP?"
        assert c1.metadata["topic"] == "NLP"
        assert c1.metadata["language"] == "en"
        assert c1.metadata["drb_prompt"] == "What are recent advances in NLP?"
        assert c1.metadata["adapter"] == "deep_research_bench"

        c2 = cases[1]
        assert c2.case_id == "drb_002"
        assert c2.metadata["language"] == "zh"

    def test_drb_prompt_preserved_verbatim(self, tmp_path):
        """The raw prompt must be preserved exactly for round-trip export."""
        prompt = "This is a complex prompt with special chars: é, ñ, 中文"
        record = {"id": "rt_001", "topic": "Test", "language": "en", "prompt": prompt}
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

        cases = DeepResearchBenchAdapter().load_cases(str(path))
        assert cases[0].metadata["drb_prompt"] == prompt

    def test_backward_compat_with_legacy_fields(self, tmp_path):
        """Legacy fields (question, field) should still work."""
        record = {"case_id": "old_001", "question": "Legacy question", "field": "cs"}
        path = tmp_path / "legacy.jsonl"
        path.write_text(json.dumps(record))

        cases = DeepResearchBenchAdapter().load_cases(str(path))
        assert cases[0].case_id == "old_001"
        assert cases[0].query == "Legacy question"

    def test_loads_sample_drb_fixture(self):
        fixture = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "evals", "fixtures", "sample_drb_queries.jsonl"
        )
        if not os.path.exists(fixture):
            pytest.skip("sample_drb_queries.jsonl not found")

        cases = DeepResearchBenchAdapter().load_cases(fixture)
        assert len(cases) == 3
        for c in cases:
            assert c.case_id.startswith("sample_drb_")
            assert c.metadata.get("topic")
            assert c.metadata.get("language") == "en"


# ---------------------------------------------------------------------------
# DRB Output Export
# ---------------------------------------------------------------------------

class TestDRBOutputReporter:

    def test_writes_official_format(self, tmp_path):
        results = [
            EvalResult(
                case_id="drb_001",
                status=RunStatus.SUCCESS,
                query="What are recent advances?",
                report_content="# Report\n\nDeep learning has...",
                case_metadata={"drb_prompt": "Official DRB prompt text", "adapter": "deep_research_bench"},
            ),
            EvalResult(
                case_id="drb_002",
                status=RunStatus.SUCCESS,
                query="Another query",
                report_content="# Another\n\nContent...",
                case_metadata={},
            ),
        ]

        reporter = DRBOutputReporter(model_name="my_model")
        path = reporter.write(results, str(tmp_path))

        assert os.path.basename(path) == "my_model.jsonl"
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 2

        # First record uses preserved DRB prompt
        assert lines[0]["id"] == "drb_001"
        assert lines[0]["prompt"] == "Official DRB prompt text"
        assert lines[0]["article"] == "# Report\n\nDeep learning has..."

        # Second record falls back to query
        assert lines[1]["id"] == "drb_002"
        assert lines[1]["prompt"] == "Another query"

    def test_handles_empty_report(self, tmp_path):
        results = [
            EvalResult(
                case_id="fail_001",
                status=RunStatus.FAILURE,
                query="Failed query",
                report_content=None,
                case_metadata={"drb_prompt": "Prompt"},
            ),
        ]
        reporter = DRBOutputReporter(model_name="test")
        path = reporter.write(results, str(tmp_path))
        with open(path) as f:
            record = json.loads(f.readline())
        assert record["article"] == ""

    def test_output_keys_match_drb_spec(self, tmp_path):
        """Output records must have exactly {id, prompt, article}."""
        results = [
            EvalResult(
                case_id="spec_001",
                status=RunStatus.SUCCESS,
                query="q",
                report_content="report",
                case_metadata={"drb_prompt": "p"},
            ),
        ]
        reporter = DRBOutputReporter(model_name="check")
        path = reporter.write(results, str(tmp_path))
        with open(path) as f:
            record = json.loads(f.readline())
        assert set(record.keys()) == {"id", "prompt", "article"}


# ---------------------------------------------------------------------------
# DRB Evaluator — Validation
# ---------------------------------------------------------------------------

class TestDRBEvaluatorValidation:

    def test_validate_repo_missing_dir(self):
        errors = _validate_drb_repo("/nonexistent/path")
        assert any("does not exist" in e for e in errors)

    def test_validate_repo_structure(self, tmp_path):
        """Should flag missing expected files."""
        errors = _validate_drb_repo(str(tmp_path))
        assert len(errors) > 0
        assert any("deepresearch_bench_race.py" in e for e in errors)

    def test_validate_repo_valid(self, tmp_path):
        """Should pass when all expected files exist."""
        (tmp_path / "deepresearch_bench_race.py").touch()
        (tmp_path / "data" / "prompt_data").mkdir(parents=True)
        (tmp_path / "data" / "prompt_data" / "query.jsonl").touch()
        (tmp_path / "utils").mkdir()

        errors = _validate_drb_repo(str(tmp_path))
        assert errors == []

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_env_vars_missing(self):
        errors = _validate_env_vars()
        # Default provider is deepseek, so should require DEEPSEEK_API_KEY
        assert any("DEEPSEEK_API_KEY" in e for e in errors)
        assert any("JINA_API_KEY" in e for e in errors)

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test", "JINA_API_KEY": "test"})
    def test_validate_env_vars_present(self):
        errors = _validate_env_vars()
        assert errors == []

    def test_resolve_from_cli_arg(self, tmp_path):
        path = resolve_drb_repo_path(str(tmp_path))
        assert path == str(tmp_path.resolve())

    @patch.dict(os.environ, {"DRB_REPO_PATH": "/some/path"})
    def test_resolve_from_env(self):
        path = resolve_drb_repo_path(None)
        assert "some/path" in path

    def test_resolve_raises_without_either(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DRB repo path not specified"):
                resolve_drb_repo_path(None)


# ---------------------------------------------------------------------------
# CLI Arg Parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:

    def test_inspect_cmd(self):
        from evals.cli import parse_args
        args = parse_args(["inspect", "--queries", "data/query.jsonl"])
        assert args.command == "inspect"
        assert args.queries == "data/query.jsonl"
        assert args.adapter == "drb"

    def test_generate_cmd(self):
        from evals.cli import parse_args
        args = parse_args([
            "generate", "--queries", "q.jsonl", "--output", "out/",
            "--model-name", "test_model", "--limit", "5",
        ])
        assert args.command == "generate"
        assert args.model_name == "test_model"
        assert args.limit == 5

    def test_evaluate_cmd(self):
        from evals.cli import parse_args
        args = parse_args([
            "evaluate", "--model-name", "my_model",
            "--drb-repo-path", "/path/to/drb",
            "--phase", "race", "--limit", "2",
        ])
        assert args.command == "evaluate"
        assert args.model_name == "my_model"
        assert args.drb_repo_path == "/path/to/drb"
        assert args.phase == "race"
        assert args.limit == 2

    def test_run_cmd_backward_compat(self):
        from evals.cli import parse_args
        args = parse_args(["run", "--cases", "cases.jsonl"])
        assert args.command == "run"
        assert args.adapter == "jsonl"
