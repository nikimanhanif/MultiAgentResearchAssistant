"""
DeepResearchBench dataset adapter.

Aligns with the official DRB public schema from:
  https://github.com/Ayanami0730/deep_research_bench

Official DRB query format (data/prompt_data/query.jsonl):
    {"id": "...", "topic": "...", "language": "en|zh", "prompt": "..."}

Official DRB output format (data/test_data/raw_data/<model>.jsonl):
    {"id": "...", "prompt": "...", "article": "..."}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from evals.adapters.base import BenchmarkAdapter
from evals.models import BenchmarkCase


class DeepResearchBenchAdapter(BenchmarkAdapter):
    """
    Adapter for the official DeepResearchBench query format.

    Reads `query.jsonl` files with {id, topic, language, prompt} records
    and maps them into BenchmarkCase objects for the evaluation runner.

    The raw `prompt` is preserved in metadata as `drb_prompt` so it can
    be written back verbatim in the DRB output export.
    """

    def load_cases(self, source: str) -> List[BenchmarkCase]:
        cases: List[BenchmarkCase] = []
        with open(source, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    cases.append(self.adapt_case(raw))
                except (json.JSONDecodeError, ValueError) as e:
                    raise ValueError(
                        f"Failed to parse DRB line {line_num} in {source}: {e}"
                    ) from e
        return cases

    def adapt_case(self, raw: Dict[str, Any]) -> BenchmarkCase:
        """
        Map official DRB fields to BenchmarkCase.

        Official fields: id, topic, language, prompt
        Legacy/custom fields (question, field, reference_report) are handled
        as fallbacks for backward compatibility.
        """
        # Official DRB uses "prompt"; legacy uses "question" or "query"
        prompt = raw.get("prompt") or raw.get("question") or raw.get("query", "")
        case_id = raw.get("id") or raw.get("case_id", "")
        if case_id == "":
             case_id = "unknown" # Fallback if no ID found

        metadata: Dict[str, Any] = {"adapter": "deep_research_bench"}

        # Preserve the raw DRB prompt for verbatim export
        metadata["drb_prompt"] = prompt

        # Official DRB fields
        if "topic" in raw:
            metadata["topic"] = raw["topic"]
        if "language" in raw:
            metadata["language"] = raw["language"]

        # Legacy fields (backward compat)
        for key in ("field", "difficulty", "source_dataset"):
            if key in raw:
                metadata[key] = raw[key]

        return BenchmarkCase(
            case_id=case_id,
            query=prompt,
            context=raw.get("context"),
            constraints=raw.get("constraints"),
            expected_answer=raw.get("reference_report") or raw.get("expected_answer"),
            expected_topics=raw.get("expected_topics"),
            metadata=metadata,
        )
