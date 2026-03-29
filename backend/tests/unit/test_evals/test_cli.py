import argparse
import os
import pytest
from evals.cli import _parse_range, _filter_cases_by_range
from evals.models import BenchmarkCase

class TestCliRangeParsing:
    def test_parse_range_valid(self):
        assert _parse_range("1-10") == (1, 10)
        assert _parse_range("5-5") == (5, 5)

    def test_parse_range_invalid_format(self):
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _parse_range("1")
        assert "Expected format" in str(exc.value)

        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _parse_range("1-10-20")
        assert "Expected format" in str(exc.value)

    def test_parse_range_invalid_types(self):
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _parse_range("a-b")
        assert "must be integers" in str(exc.value)

    def test_parse_range_invalid_values(self):
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _parse_range("0-10")
        assert "START must be >= 1" in str(exc.value)

        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _parse_range("10-5")
        assert "START (10) must be <= END (5)" in str(exc.value)

    def test_filter_cases_by_range(self):
        cases = [
            BenchmarkCase(case_id="1", query="q1"),
            BenchmarkCase(case_id="2", query="q2"),
            BenchmarkCase(case_id="3", query="q3"),
            BenchmarkCase(case_id="alpha", query="q4"),  # Non-numeric
        ]
        
        filtered = _filter_cases_by_range(cases, 2, 3)
        assert len(filtered) == 2
        assert filtered[0].case_id == "2"
        assert filtered[1].case_id == "3"
        
        # Non-numeric should be safely ignored
        filtered_all = _filter_cases_by_range(cases, 1, 10)
        assert len(filtered_all) == 3

class TestStatusAndCollision:
    def test_cmd_status_reports_missing(self, tmp_path, caplog):
        import json
        import asyncio
        from evals.cli import cmd_status

        out_dir = str(tmp_path)
        model_name = "test_model"
        out_path = os.path.join(out_dir, f"{model_name}.jsonl")

        # Create mock output holding sample_002
        with open(out_path, "w") as f:
            f.write(json.dumps({"id": "sample_002", "article": "done"}) + "\n")

        args = argparse.Namespace(
            adapter="jsonl",
            queries="evals/fixtures/sample_cases.jsonl",
            output=out_dir,
            model_name=model_name
        )
        
        with caplog.at_level("INFO"):
            asyncio.run(cmd_status(args))
            
        # sample_cases has sample_001 through sample_003
        assert "Completed: 1/3" in caplog.text
        assert "Missing IDs (2): sample_001, sample_003" in caplog.text

    def test_cmd_generate_skips_existing_by_default(self, tmp_path, caplog):
        import json
        import asyncio
        from evals.cli import cmd_generate

        out_dir = str(tmp_path)
        model_name = "test_model"
        out_path = os.path.join(out_dir, f"{model_name}.jsonl")

        # Suppose sample_002 is already done
        with open(out_path, "w") as f:
            f.write(json.dumps({"id": "sample_002", "article": "done"}) + "\n")

        # We ask to run sample_002. It should be skipped due to collision filtering.
        args = argparse.Namespace(
            adapter="jsonl",
            queries="evals/fixtures/sample_cases.jsonl",
            output=out_dir,
            model_name=model_name,
            case_id="sample_002",
            range_tuple=None,
            force=False,
            dry_run=True,
            limit=None,
            shuffle=False
        )
        
        with caplog.at_level("INFO"):
            asyncio.run(cmd_generate(args))
            
        assert "Skipping 1 cases already present in output file: sample_002" in caplog.text
        assert "No new cases to run. Exiting" in caplog.text

    def test_cmd_generate_force_overwrites(self, tmp_path, caplog):
        import json
        import asyncio
        from evals.cli import cmd_generate

        out_dir = str(tmp_path)
        model_name = "test_model"
        out_path = os.path.join(out_dir, f"{model_name}.jsonl")

        # Suppose sample_002 is already done
        with open(out_path, "w") as f:
            f.write(json.dumps({"id": "sample_002", "article": "done"}) + "\n")

        # We ask to run sample_002 with force=True. It should NOT be skipped.
        args = argparse.Namespace(
            adapter="jsonl",
            queries="evals/fixtures/sample_cases.jsonl",
            output=out_dir,
            model_name=model_name,
            case_id="sample_002",
            range_tuple=None,
            force=True,
            dry_run=True,
            limit=None,
            shuffle=False
        )
        
        with caplog.at_level("INFO"):
            asyncio.run(cmd_generate(args))
            
        assert "Skipping" not in caplog.text
        assert "Dry run complete." in caplog.text
