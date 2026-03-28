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
    _add_budget_args(gen_parser)

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

    if args.case_id:
        # Match as string to be safe, but handle potential int IDs
        cases = [c for c in cases if str(c.case_id) == args.case_id]
        if not cases:
            logger.error("Case ID '%s' not found in %s", args.case_id, args.queries)
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

    if args.case_id:
        cases = [c for c in cases if str(c.case_id) == args.case_id]
        if not cases:
            logger.error("Case ID '%s' not found in %s", args.case_id, args.queries)
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

    # Write DRB-format output
    drb_reporter = DRBOutputReporter(model_name=args.model_name)
    drb_path = drb_reporter.write(results, args.output)
    logger.info("DRB output written: %s", drb_path)

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
# Main
# ---------------------------------------------------------------------------

COMMAND_MAP = {
    "inspect": cmd_inspect,
    "generate": cmd_generate,
    "evaluate": cmd_evaluate,
    "run": cmd_run,
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
