"""
Generic JSONL adapter for loading benchmark cases.

Expects one JSON object per line matching the BenchmarkCase schema.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from evals.adapters.base import BenchmarkAdapter
from evals.models import BenchmarkCase


class JSONLAdapter(BenchmarkAdapter):
    """Reads BenchmarkCase objects from a JSONL file."""

    def load_cases(self, source: str) -> List[BenchmarkCase]:
        cases: List[BenchmarkCase] = []
        with open(source, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    cases.append(self.adapt_case(raw))
                except (json.JSONDecodeError, ValueError) as e:
                    raise ValueError(
                        f"Failed to parse line {line_num} in {source}: {e}"
                    ) from e
        return cases
