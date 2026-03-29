"""
Result reporting / export layer.

Supports JSONL (internal results), CSV summary, Markdown report,
and official DRB output format ({id, prompt, article}).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from evals.models import EvalResult

logger = logging.getLogger(__name__)


class Reporter(ABC):
    """Protocol for result reporters."""

    @abstractmethod
    def write(self, results: List[EvalResult], output_dir: str) -> str:
        """Write results to output_dir. Returns path of the written file."""
        ...


class JSONLReporter(Reporter):
    """Writes one JSON object per result, one per line."""

    def write(self, results: List[EvalResult], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "results.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")
        return path


class CSVReporter(Reporter):
    """Writes a flat CSV with one row per case and key metrics as columns."""

    def write(self, results: List[EvalResult], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "summary.csv")

        # Collect all unique metric names across results
        all_metrics: list[str] = []
        seen: set[str] = set()
        for r in results:
            for s in r.scores:
                if s.metric not in seen:
                    all_metrics.append(s.metric)
                    seen.add(s.metric)

        fieldnames = [
            "case_id", "status", "query", "findings_count",
            "duration_seconds", "clarification_interrupts", "review_interrupts",
        ] + all_metrics

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {
                    "case_id": r.case_id,
                    "status": r.status.value,
                    "query": r.query[:100],
                    "findings_count": r.findings_count,
                    "duration_seconds": r.metadata.duration_seconds,
                    "clarification_interrupts": r.metadata.clarification_interrupts,
                    "review_interrupts": r.metadata.review_interrupts,
                }
                score_map = {s.metric: s.value for s in r.scores}
                for m in all_metrics:
                    row[m] = score_map.get(m, "")
                writer.writerow(row)
        return path


class MarkdownReporter(Reporter):
    """Generates a human-readable Markdown summary report."""

    def write(self, results: List[EvalResult], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "report.md")

        lines: list[str] = []
        lines.append("# Evaluation Report\n")
        lines.append(f"**Cases run:** {len(results)}\n")

        success_count = sum(1 for r in results if r.status.value == "success")
        lines.append(f"**Successful:** {success_count}/{len(results)}\n")

        # Aggregate metrics
        all_metrics: dict[str, list[float]] = {}
        for r in results:
            for s in r.scores:
                all_metrics.setdefault(s.metric, []).append(s.value)

        if all_metrics:
            lines.append("\n## Aggregate Scores\n")
            lines.append("| Metric | Mean | Min | Max |")
            lines.append("|--------|------|-----|-----|")
            for metric, values in all_metrics.items():
                mean = sum(values) / len(values)
                lines.append(
                    f"| {metric} | {mean:.4f} | {min(values):.4f} | {max(values):.4f} |"
                )

        lines.append("\n## Per-Case Results\n")
        for r in results:
            lines.append(f"### {r.case_id}\n")
            lines.append(f"- **Status:** {r.status.value}")
            lines.append(f"- **Query:** {r.query[:120]}")
            lines.append(f"- **Findings:** {r.findings_count}")
            dur = r.metadata.duration_seconds
            lines.append(f"- Duration: {dur:.1f}s" if dur is not None else "- Duration: N/A")
            if r.scores:
                lines.append("- **Scores:**")
                for s in r.scores:
                    detail = f" ({s.detail})" if s.detail else ""
                    lines.append(f"  - {s.metric}: {s.value}{detail}")
            if r.metadata.error_log:
                lines.append("- **Errors:**")
                for err in r.metadata.error_log:
                    lines.append(f"  - {err[:200]}")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path


class DRBOutputReporter(Reporter):
    """
    Writes model outputs in the official DRB format expected by run_benchmark.sh.

    Each line: {"id": "...", "prompt": "...", "article": "..."}

    The `prompt` is the original DRB prompt preserved verbatim from the adapter.
    The `article` is the report generated by our research pipeline.
    """

    def __init__(self, model_name: str = "model"):
        self.model_name = model_name

    def write(self, results: List[EvalResult], output_dir: str, *, append: bool = False) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.model_name}.jsonl")

        # Build new records from this run
        new_records: dict[str | int, dict] = {}
        for r in results:
            drb_prompt = (r.case_metadata or {}).get("drb_prompt", r.query)
            # Normalize numeric string IDs to int — DRB's query.jsonl uses integer IDs
            # and extract.py builds id_to_lang_map with int keys, so a string "51" would
            # fail the lookup and produce "Language not found" errors.
            case_id = int(r.case_id) if isinstance(r.case_id, str) and r.case_id.isdigit() else r.case_id
            record = {
                "id": case_id,
                "prompt": drb_prompt,
                "article": r.report_content or "",
            }
            new_records[case_id] = record

        if append and os.path.exists(path):
            # Load existing entries, keyed by id
            existing: dict[str | int, dict] = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if "id" in entry:
                            existing[entry["id"]] = entry
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed or incomplete JSON line in %s", path)
                        continue
            # Merge: new results overwrite matching IDs
            existing.update(new_records)
            merged = existing
        else:
            merged = new_records

        # Sort by numeric id (fall back to string sort for non-numeric)
        def _sort_key(entry: dict):
            eid = entry["id"]
            return (0, int(eid)) if isinstance(eid, int) or (isinstance(eid, str) and eid.isdigit()) else (1, str(eid))

        sorted_entries = sorted(merged.values(), key=_sort_key)

        with open(path, "w", encoding="utf-8") as f:
            for entry in sorted_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path
