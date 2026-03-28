"""
CLI entrypoint for the evaluation harness.

Commands:
    inspect   — Preview loaded benchmark cases without running the graph
    generate  — Run the harness and export DRB-compatible output JSONL
    evaluate  — Invoke official DRB RACE + FACT evaluation scripts
    run       — Run internal evaluation (legacy harness with internal scorers)

Examples:
    # Inspect DRB queries
    python -m evals.cli inspect --queries /path/to/query.jsonl

    # Generate DRB-format outputs (smoke run, 3 cases)
    python -m evals.cli generate --queries /path/to/query.jsonl --output results/ --model-name my_model --limit 3

    # Dry-run generate (validate without running graph)
    python -m evals.cli generate --queries /path/to/query.jsonl --output results/ --model-name my_model --dry-run

    # Run official DRB evaluation
    python -m evals.cli evaluate --drb-repo-path /path/to/deep_research_bench --model-name my_model

    # Internal evaluation (legacy)
    python -m evals.cli run --cases evals/fixtures/sample_cases.jsonl --output results/
"""

from __future__ import annotations

import argparse
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env to os.environ for external tools
load_dotenv()

import logging
import os
import sys
from typing import Optional

from evals.adapters.jsonl_adapter import JSONLAdapter
from evals.adapters.deep_research_bench import DeepResearchBenchAdapter
from evals.models import BenchmarkCase
from evals.policies import (
    DefaultClarificationPolicy,
    AutoApproveReviewPolicy,
    DefaultFailurePolicy,
)
from evals.reporters import (
    JSONLReporter,
    CSVReporter,
    DRBOutputReporter,
    MarkdownReporter,
    Reporter,
)
from evals.runner import EvalRunner
from evals.scorers import (
    CitationScorer,
    CompletenessScorer,
    FindingScorer,
    LatencyScorer,
    Scorer,
    SuccessScorer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ADAPTER_MAP = {
    "jsonl": JSONLAdapter,
    "drb": DeepResearchBenchAdapter,
}

REPORTER_MAP: dict[str, type[Reporter]] = {
    "jsonl": JSONLReporter,
    "csv": CSVReporter,
    "markdown": MarkdownReporter,
}

INTERNAL_SCORERS: list[Scorer] = [
    SuccessScorer(),
    LatencyScorer(),
    FindingScorer(),
    CitationScorer(),
    CompletenessScorer(),
]


def _parse_range(range_str: str) -> tuple[int, int]:
    """Parse a 'START-END' range string. Both inclusive, 1-indexed."""
    parts = range_str.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid range '{range_str}'. Expected format: START-END (e.g., 1-10)"
        )
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid range '{range_str}'. START and END must be integers."
        )
    if start < 1:
        raise argparse.ArgumentTypeError(
            f"Invalid range '{range_str}'. START must be >= 1."
        )
    if start > end:
        raise argparse.ArgumentTypeError(
            f"Invalid range '{range_str}'. START ({start}) must be <= END ({end})."
        )
    return start, end


def _filter_cases_by_range(
    cases: list[BenchmarkCase], start: int, end: int,
) -> list[BenchmarkCase]:
    """Filter cases whose numeric case_id falls within [start, end] inclusive."""
    filtered = []
    for c in cases:
        try:
            cid = int(c.case_id)
        except (ValueError, TypeError):
            continue
        if start <= cid <= end:
            filtered.append(c)
    return filtered


def _add_range_arg(parser: argparse.ArgumentParser) -> None:
    """Add shared --range argument."""
    parser.add_argument(
        "--range", type=_parse_range, default=None, dest="range_tuple",
        help="Run a range of cases by ID: START-END (inclusive, e.g., 1-10)",
    )


def _add_budget_args(parser: argparse.ArgumentParser) -> None:
    """Shared budget override arguments."""
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Override max_iterations budget for shorter runs",
    )
    parser.add_argument(
        "--max-sub-agents", type=int, default=None,
        help="Override max_sub_agents budget",
    )


def _build_budget_overrides(args: argparse.Namespace) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if getattr(args, "max_iterations", None) is not None:
        overrides["max_iterations"] = args.max_iterations
    if getattr(args, "max_sub_agents", None) is not None:
        overrides["max_sub_agents"] = args.max_sub_agents
    return overrides


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evals",
        description="Evaluation harness for the production research graph.",
    )
    sub = parser.add_subparsers(dest="command")

    # ---- inspect ----
    inspect_parser = sub.add_parser(
        "inspect",
        help="Preview loaded benchmark cases without running the graph",
    )
    inspect_parser.add_argument(
        "--queries", required=True,
        help="Path to benchmark query file (JSONL). Use DRB query.jsonl or custom file.",
    )
    inspect_parser.add_argument(
        "--adapter", choices=list(ADAPTER_MAP.keys()), default="drb",
        help="Dataset adapter (default: drb)",
    )
    inspect_parser.add_argument(
        "--limit", type=int, default=None,
        help="Show only the first N cases",
    )
    inspect_parser.add_argument(
        "--shuffle", action="store_true",
        help="Randomly shuffle cases before applying limit",
    )
    inspect_parser.add_argument(
        "--case-id", type=str, default=None,
        help="Target a specific case by its ID",
    )
    _add_range_arg(inspect_parser)

    # ---- generate ----
    gen_parser = sub.add_parser(
        "generate",
        help="Run harness and export DRB-compatible output JSONL",
    )
    gen_parser.add_argument(
        "--queries", required=True,
        help="Path to benchmark query file (JSONL)",
    )
    gen_parser.add_argument(
        "--output", default="eval_results",
        help="Output directory for the generated JSONL and internal reports",
    )
    gen_parser.add_argument(
        "--model-name", default="multi_agent_researcher",
        help="Model name for the output file (becomes <model-name>.jsonl)",
    )
    gen_parser.add_argument(
        "--adapter", choices=list(ADAPTER_MAP.keys()), default="drb",
        help="Dataset adapter (default: drb)",
    )
    gen_parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N cases (useful for smoke tests)",
    )
    gen_parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate cases without running the graph",
    )
    gen_parser.add_argument(
        "--shuffle", action="store_true",
        help="Randomly shuffle cases before applying limit",
    )
    gen_parser.add_argument(
        "--case-id", type=str, default=None,
        help="Target a specific case by its ID",
    )
    gen_parser.add_argument(
        "--force", action="store_true",
        help="Force overwrite existing cases in the output file",
    )
    _add_range_arg(gen_parser)
    _add_budget_args(gen_parser)
    
    # ---- status ----
    status_parser = sub.add_parser(
        "status",
        help="Check the completion status of the benchmark against the output file",
    )
    status_parser.add_argument(
        "--queries", required=True,
        help="Path to benchmark query file (JSONL)",
    )
    status_parser.add_argument(
        "--output", default="eval_results",
        help="Output directory where the JSONL results are stored",
    )
    status_parser.add_argument(
        "--model-name", default="multi_agent_researcher",
        help="Model name for the output file (e.g., <model-name>.jsonl)",
    )
    status_parser.add_argument(
        "--adapter", choices=list(ADAPTER_MAP.keys()), default="drb",
        help="Dataset adapter (default: drb)",
    )

    # ---- evaluate ----
    eval_parser = sub.add_parser(
        "evaluate",
        help="Run official DRB RACE + FACT evaluation against generated outputs",
    )
    eval_parser.add_argument(
        "--drb-repo-path", default=None,
        help="Path to local clone of deep_research_bench repo (or set DRB_REPO_PATH env var)",
    )
    eval_parser.add_argument(
        "--model-name", required=True,
        help="Model name matching the output file in raw_data/",
    )
    eval_parser.add_argument(
        "--raw-output-path", default=None,
        help="Path to raw model output JSONL (if different from DRB default location)",
    )
    eval_parser.add_argument(
        "--max-workers", type=int, default=10,
        help="Number of parallel workers for DRB evaluation",
    )
    eval_parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of prompts to process (for testing)",
    )
    eval_parser.add_argument(
        "--case-id", type=str, default=None,
        help="Target a specific case by its ID",
    )
    eval_parser.add_argument(
        "--only-language", choices=["en", "zh"], default=None,
        help="Only process a specific language",
    )
    eval_parser.add_argument(
        "--skip-cleaning", action="store_true",
        help="Skip article cleaning step in RACE",
    )
    eval_parser.add_argument(
        "--force", action="store_true",
        help="Force re-evaluation even if results exist",
    )
    eval_parser.add_argument(
        "--phase", choices=["race", "fact", "both"], default="both",
        help="Which evaluation phase to run (default: both)",
    )

    # ---- run (legacy internal) ----
    run_parser = sub.add_parser(
        "run",
        help="Run internal evaluation harness (internal scorers, not official DRB)",
    )
    run_parser.add_argument("--cases", required=True, help="Path to benchmark cases JSONL")
    run_parser.add_argument("--output", default="eval_results", help="Output directory")
    run_parser.add_argument(
        "--format", nargs="+", choices=list(REPORTER_MAP.keys()),
        default=["jsonl", "csv", "markdown"], help="Output formats",
    )
    run_parser.add_argument(
        "--adapter", choices=list(ADAPTER_MAP.keys()), default="jsonl",
        help="Dataset adapter (default: jsonl)",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Validate only")
    run_parser.add_argument("--shuffle", action="store_true", help="Randomly shuffle cases")
    run_parser.add_argument("--case-id", type=str, default=None, help="Target specific case ID")
    _add_budget_args(run_parser)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Command: inspect
# ---------------------------------------------------------------------------

async def cmd_inspect(args: argparse.Namespace) -> None:
    adapter = ADAPTER_MAP[args.adapter]()
    cases = adapter.load_cases(args.queries)
    total = len(cases)

    if args.case_id and getattr(args, "range_tuple", None):
        logger.error("--case-id and --range are mutually exclusive.")
        return

    if args.case_id:
        cases = [c for c in cases if str(c.case_id) == args.case_id]
        if not cases:
            logger.error("Case ID '%s' not found in %s", args.case_id, args.queries)
            return

    if getattr(args, "range_tuple", None):
        start, end = args.range_tuple
        cases = _filter_cases_by_range(cases, start, end)
        if not cases:
            logger.error("No cases found in range %d-%d", start, end)
            return

    if args.shuffle:
        import random
        random.shuffle(cases)

    if args.limit:
        cases = cases[:args.limit]

    logger.info("Loaded %d total cases (showing %d)", total, len(cases))
    for c in cases:
        topic = c.metadata.get("topic", "N/A")
        lang = c.metadata.get("language", "N/A")
        logger.info(
            "  [%s] topic=%s lang=%s query=%.80s",
            c.case_id, topic, lang, c.query,
        )
    logger.info("Inspection complete. %d cases available.", total)


# ---------------------------------------------------------------------------
# Command: generate
# ---------------------------------------------------------------------------

async def cmd_generate(args: argparse.Namespace) -> None:
    adapter = ADAPTER_MAP[args.adapter]()
    cases = adapter.load_cases(args.queries)
    total = len(cases)
    use_append = False

    if args.case_id and getattr(args, "range_tuple", None):
        logger.error("--case-id and --range are mutually exclusive.")
        return

    if args.case_id:
        cases = [c for c in cases if str(c.case_id) == args.case_id]
        if not cases:
            logger.error("Case ID '%s' not found in %s", args.case_id, args.queries)
            return
        use_append = True

    if getattr(args, "range_tuple", None):
        start, end = args.range_tuple
        cases = _filter_cases_by_range(cases, start, end)
        if not cases:
            logger.error("No cases found in range %d-%d", start, end)
            return
        use_append = True
        logger.info("Range mode: cases %d-%d (append enabled)", start, end)

    # Collision detection
    if use_append and not args.force:
        output_path = os.path.join(args.output, f"{args.model_name}.jsonl")
        if os.path.exists(output_path):
            existing_ids = set()
            import json
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        entry = json.loads(line)
                        if "id" in entry:
                            existing_ids.add(str(entry["id"]))
                    except json.JSONDecodeError:
                        continue
            
            filtered_cases = []
            skipped_ids = []
            for c in cases:
                if str(c.case_id) in existing_ids:
                    skipped_ids.append(str(c.case_id))
                else:
                    filtered_cases.append(c)
                    
            if skipped_ids:
                logger.warning(
                    "Skipping %d cases already present in output file: %s. Use --force to overwrite.",
                    len(skipped_ids), ", ".join(skipped_ids[:10]) + ("..." if len(skipped_ids) > 10 else "")
                )
            cases = filtered_cases
            if not cases:
                logger.info("No new cases to run. Exiting.")
                return

    if args.shuffle:
        import random
        random.shuffle(cases)

    if args.limit:
        cases = cases[:args.limit]

    logger.info("Loaded %d cases (running %d)", total, len(cases))

    if args.dry_run:
        for c in cases:
            logger.info("  [%s] %.80s", c.case_id, c.query)
        logger.info("Dry run complete. No graph execution performed.")
        return

    runner = EvalRunner(
        scorers=INTERNAL_SCORERS,
        budget_overrides=_build_budget_overrides(args),
    )

    def _on_result(result):
        logger.info(
            "  [%s] status=%s findings=%d duration=%.1fs",
            result.case_id, result.status.value,
            result.findings_count,
            result.metadata.duration_seconds or 0,
        )

    results = await runner.run_batch(cases, on_result=_on_result)

    # Write DRB-format output (append mode for --range and --case-id)
    drb_reporter = DRBOutputReporter(model_name=args.model_name)
    drb_path = drb_reporter.write(results, args.output, append=use_append)
    logger.info("DRB output written: %s%s", drb_path, " (merged)" if use_append else "")

    # Also write internal reports
    for fmt_name, ReporterCls in REPORTER_MAP.items():
        r = ReporterCls()
        p = r.write(results, args.output)
        logger.info("Internal %s report: %s", fmt_name, p)

    successes = sum(1 for r in results if r.status.value == "success")
    logger.info(
        "Done. %d/%d cases succeeded. DRB output: %s",
        successes, len(results), drb_path,
    )


# ---------------------------------------------------------------------------
# Command: evaluate
# ---------------------------------------------------------------------------

async def cmd_evaluate(args: argparse.Namespace) -> None:
    from evals.drb_evaluator import (
        DRBEvalConfig,
        resolve_drb_repo_path,
        run_full_evaluation,
        setup_drb_judge,
        run_race,
        run_fact,
        _get_judge_provider,
    )

    drb_path = resolve_drb_repo_path(args.drb_repo_path)
    provider = _get_judge_provider()
    logger.info("Using DRB repo: %s (judge: %s)", drb_path, provider)

    config = DRBEvalConfig(
        drb_repo_path=drb_path,
        model_name=args.model_name,
        raw_output_path=args.raw_output_path,
        max_workers=args.max_workers,
        limit=args.limit,
        case_id=args.case_id,
        only_language=args.only_language,
        skip_cleaning=args.skip_cleaning,
        force=args.force,
    )

    if args.phase in ("race", "fact"):
        # For individual phases, deploy judge first
        setup_drb_judge(drb_path)
        if args.phase == "race":
            rc = run_race(config)
            logger.info("RACE evaluation finished (exit code: %d, judge: %s)", rc, provider)
        else:
            rc = run_fact(config)
            logger.info("FACT evaluation finished (exit code: %d, judge: %s)", rc, provider)
    else:
        result = run_full_evaluation(config)
        if result.success:
            logger.info("DRB evaluation completed successfully! (judge: %s)", result.judge_provider)
            if result.race_result_path:
                logger.info("RACE results: %s", result.race_result_path)
            if result.fact_result_path:
                logger.info("FACT results: %s", result.fact_result_path)
        else:
            logger.error("DRB evaluation had errors:")
            for err in result.errors:
                logger.error("  %s", err)
            if result.race_return_code and result.race_return_code != 0:
                logger.error("  RACE exited with code %d", result.race_return_code)
            if result.fact_return_code and result.fact_return_code != 0:
                logger.error("  FACT exited with code %d", result.fact_return_code)


# ---------------------------------------------------------------------------
# Command: run (legacy internal)
# ---------------------------------------------------------------------------

async def cmd_run(args: argparse.Namespace) -> None:
    adapter = ADAPTER_MAP[args.adapter]()
    cases = adapter.load_cases(args.cases)
    
    if args.case_id:
        cases = [c for c in cases if str(c.case_id) == args.case_id]
        if not cases:
            logger.error("Case ID '%s' not found", args.case_id)
            return

    if args.shuffle:
        import random
        random.shuffle(cases)
        
    logger.info("Loaded %d benchmark cases", len(cases))

    if args.dry_run:
        for c in cases:
            logger.info("  [%s] %.80s", c.case_id, c.query)
        logger.info("Dry run complete.")
        return

    runner = EvalRunner(
        scorers=INTERNAL_SCORERS,
        budget_overrides=_build_budget_overrides(args),
    )

    def _on_result(result):
        logger.info(
            "  [%s] status=%s findings=%d duration=%.1fs",
            result.case_id, result.status.value,
            result.findings_count,
            result.metadata.duration_seconds or 0,
        )

    results = await runner.run_batch(cases, on_result=_on_result)

    for fmt in args.format:
        reporter = REPORTER_MAP[fmt]()
        path = reporter.write(results, args.output)
        logger.info("Wrote %s report: %s", fmt, path)

    successes = sum(1 for r in results if r.status.value == "success")
    logger.info("Done. %d/%d cases succeeded.", successes, len(results))


# ---------------------------------------------------------------------------
# Command: status
# ---------------------------------------------------------------------------

async def cmd_status(args: argparse.Namespace) -> None:
    adapter = ADAPTER_MAP[args.adapter]()
    cases = adapter.load_cases(args.queries)
    total_expected = len(cases)
    
    output_path = os.path.join(args.output, f"{args.model_name}.jsonl")
    
    if not os.path.exists(output_path):
        logger.info("Output file %s does not exist.", output_path)
        logger.info("Completed: 0/%d", total_expected)
        return
        
    existing_ids = set()
    import json
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                if "id" in entry:
                    existing_ids.add(str(entry["id"]))
            except json.JSONDecodeError:
                continue
                
    missing_ids = []
    completed_ids = []
    
    for c in cases:
        cid = str(c.case_id)
        if cid in existing_ids:
            completed_ids.append(cid)
        else:
            missing_ids.append(cid)
            
    # Sort for prettier output
    def try_int(x):
        return int(x) if x.isdigit() else x
        
    missing_ids.sort(key=lambda x: (0, try_int(x)) if x.isdigit() else (1, str(x)))
    
    logger.info("Benchmark Status: %s", output_path)
    logger.info("Completed: %d/%d", len(completed_ids), total_expected)
    if missing_ids:
        logger.info("Missing IDs (%d): %s", len(missing_ids), ", ".join(missing_ids))
    else:
        logger.info("All cases completed!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COMMAND_MAP = {
    "inspect": cmd_inspect,
    "generate": cmd_generate,
    "evaluate": cmd_evaluate,
    "run": cmd_run,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    handler = COMMAND_MAP.get(args.command)
    if handler:
        asyncio.run(handler(args))
    else:
        parse_args(["--help"])


if __name__ == "__main__":
    main()
