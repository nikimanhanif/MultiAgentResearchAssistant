"""
Unit tests for evals.adapters — dataset loading and field mapping.
"""

import json
import os
import tempfile

import pytest

from evals.adapters.jsonl_adapter import JSONLAdapter
from evals.adapters.deep_research_bench import DeepResearchBenchAdapter
from evals.models import BenchmarkCase


# ---------------------------------------------------------------------------
# JSONLAdapter
# ---------------------------------------------------------------------------

class TestJSONLAdapter:

    def test_loads_valid_cases(self, tmp_path):
        data = [
            {"case_id": "a1", "query": "test query 1"},
            {"case_id": "a2", "query": "test query 2", "context": "extra context"},
        ]
        path = tmp_path / "cases.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        adapter = JSONLAdapter()
        cases = adapter.load_cases(str(path))

        assert len(cases) == 2
        assert cases[0].case_id == "a1"
        assert cases[1].context == "extra context"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text('{"case_id": "b1", "query": "q"}\n\n{"case_id": "b2", "query": "q2"}\n')

        cases = JSONLAdapter().load_cases(str(path))
        assert len(cases) == 2

    def test_raises_on_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n")

        with pytest.raises(ValueError, match="Failed to parse line 1"):
            JSONLAdapter().load_cases(str(path))

    def test_raises_on_missing_required_field(self, tmp_path):
        path = tmp_path / "incomplete.jsonl"
        path.write_text('{"case_id": "x"}\n')

        with pytest.raises(ValueError):
            JSONLAdapter().load_cases(str(path))

    def test_loads_sample_fixtures(self):
        fixtures_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "evals", "fixtures", "sample_cases.jsonl"
        )
        if not os.path.exists(fixtures_path):
            pytest.skip("sample_cases.jsonl not found")

        cases = JSONLAdapter().load_cases(fixtures_path)
        assert len(cases) >= 1
        for c in cases:
            assert c.case_id
            assert c.query


# ---------------------------------------------------------------------------
# DeepResearchBenchAdapter
# ---------------------------------------------------------------------------

class TestDeepResearchBenchAdapter:

    def test_maps_drb_fields(self, tmp_path):
        """Primary test: official DRB schema {id, topic, language, prompt}."""
        drb_record = {
            "id": "drb_001",
            "topic": "Physics",
            "language": "en",
            "prompt": "What are recent advances in quantum error correction?",
        }
        path = tmp_path / "drb.jsonl"
        path.write_text(json.dumps(drb_record) + "\n")

        adapter = DeepResearchBenchAdapter()
        cases = adapter.load_cases(str(path))

        assert len(cases) == 1
        case = cases[0]
        assert case.case_id == "drb_001"
        assert case.query == "What are recent advances in quantum error correction?"
        assert case.metadata["topic"] == "Physics"
        assert case.metadata["language"] == "en"
        assert case.metadata["drb_prompt"] == "What are recent advances in quantum error correction?"
        assert case.metadata["adapter"] == "deep_research_bench"

    def test_falls_back_to_generic_fields(self, tmp_path):
        """If DRB-specific fields are missing, it should still work with generic fields."""
        generic_record = {
            "case_id": "gen_001",
            "query": "generic question",
            "expected_answer": "generic answer",
        }
        path = tmp_path / "generic.jsonl"
        path.write_text(json.dumps(generic_record) + "\n")

        adapter = DeepResearchBenchAdapter()
        cases = adapter.load_cases(str(path))

        assert cases[0].case_id == "gen_001"
        assert cases[0].query == "generic question"
        assert cases[0].expected_answer == "generic answer"

    def test_adapt_case_directly(self):
        adapter = DeepResearchBenchAdapter()
        case = adapter.adapt_case({
            "id": "direct_1",
            "question": "Direct test",
            "field": "cs",
        })
        assert case.case_id == "direct_1"
        assert case.query == "Direct test"
        assert case.metadata["field"] == "cs"
