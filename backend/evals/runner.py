"""
Evaluation runner — orchestrates the production graph for benchmark cases.

Handles the full lifecycle: graph setup, interrupt/resume loop, state capture,
scoring, and result assembly. Uses an in-memory checkpointer to avoid
polluting production databases.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from evals.models import (
    BenchmarkCase,
    EvalResult,
    RunMetadata,
    RunStatus,
    ScoreResult,
)
from evals.policies import (
    AutoApproveReviewPolicy,
    BaseClarificationPolicy,
    BaseFailurePolicy,
    BaseReviewPolicy,
    DefaultClarificationPolicy,
    DefaultFailurePolicy,
)
from evals.scorers import Scorer

logger = logging.getLogger(__name__)

# Maximum number of interrupt/resume cycles before forcing termination.
MAX_INTERRUPT_CYCLES = 10


class EvalRunner:
    """
    Runs benchmark cases through the real production research graph.

    Uses the same build_research_graph() and Command(resume=...) mechanism
    as the production API, but with an in-memory checkpointer and
    deterministic interrupt policies.
    """

    def __init__(
        self,
        clarification_policy: Optional[BaseClarificationPolicy] = None,
        review_policy: Optional[BaseReviewPolicy] = None,
        failure_policy: Optional[BaseFailurePolicy] = None,
        scorers: Optional[List[Scorer]] = None,
        budget_overrides: Optional[Dict[str, int]] = None,
    ):
        self.clarification_policy = clarification_policy or DefaultClarificationPolicy()
        self.review_policy = review_policy or AutoApproveReviewPolicy()
        self.failure_policy = failure_policy or DefaultFailurePolicy()
        self.scorers = scorers or []
        self.budget_overrides = budget_overrides or {}

    def _build_input(self, case: BenchmarkCase) -> Dict[str, Any]:
        """Convert a BenchmarkCase into the graph's expected input dict."""
        budget = {
            "iterations": 0,
            "max_iterations": 20,
            "max_sub_agents": 20,
            "max_searches_per_agent": 2,
            "total_searches": 0,
        }
        budget.update(self.budget_overrides)
        return {
            "messages": [{"role": "user", "content": case.query}],
            "budget": budget,
        }

    async def run_case(self, case: BenchmarkCase) -> EvalResult:
        """
        Execute a single benchmark case through the production graph.

        Sets up an isolated in-memory checkpointer, runs the graph with
        the interrupt/resume loop, captures state, and scores the result.
        """
        run_meta = RunMetadata(start_time=datetime.utcnow())
        thread_id = f"eval_{case.case_id}_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id, "user_id": "eval_harness"}}

        # Isolated in-memory checkpointer for this run
        conn = await aiosqlite.connect(":memory:", isolation_level=None)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

        cp_module = None
        original_cp = None

        try:
            import app.persistence.checkpointer as cp_module
            original_cp = cp_module._checkpointer
            cp_module._checkpointer = checkpointer

            from app.graphs.research_graph import build_research_graph
            graph = build_research_graph()

            input_data: Any = self._build_input(case)
            cycle = 0

            while cycle < MAX_INTERRUPT_CYCLES:
                cycle += 1
                async for _event in graph.astream(
                    input_data, config, stream_mode="updates"
                ):
                    pass

                state = await graph.aget_state(config)
                next_nodes = list(state.next) if state.next else []

                if not next_nodes:
                    break

                if "scope_wait" in next_nodes:
                    run_meta.clarification_interrupts += 1
                    resume_value = self.clarification_policy.respond(case, state.values)
                    input_data = Command(resume=resume_value)
                elif "reviewer" in next_nodes:
                    run_meta.review_interrupts += 1
                    resume_value = self.review_policy.respond(state.values)
                    input_data = Command(resume=resume_value)
                else:
                    logger.warning(
                        "Unexpected pending nodes %s for case %s, stopping.",
                        next_nodes, case.case_id,
                    )
                    break

            # Capture final state
            final_state = await graph.aget_state(config)
            values = final_state.values if final_state else {}

            run_meta.end_time = datetime.utcnow()
            run_meta.duration_seconds = (
                run_meta.end_time - run_meta.start_time
            ).total_seconds()
            run_meta.finding_count = len(values.get("findings", []))
            run_meta.graph_iterations = values.get("budget", {}).get("iterations", 0)
            run_meta.error_log = values.get("error", [])

            report = values.get("report_content")
            is_complete = values.get("is_complete", False)

            status = RunStatus.SUCCESS if (is_complete and report) else RunStatus.PARTIAL

            result = EvalResult(
                case_id=case.case_id,
                status=status,
                query=case.query,
                report_content=report,
                findings_count=run_meta.finding_count,
                metadata=run_meta,
                case_metadata=case.metadata,
            )

            # Apply scorers
            scores: List[ScoreResult] = []
            for scorer in self.scorers:
                scores.extend(scorer.score(result, case, values))
            result.scores = scores

            return result

        except Exception as exc:
            run_meta.end_time = datetime.utcnow()
            run_meta.duration_seconds = (
                run_meta.end_time - run_meta.start_time
            ).total_seconds()
            logger.exception("Eval run failed for case %s", case.case_id)

            partial_state = None
            try:
                s = await graph.aget_state(config)
                partial_state = s.values if s else None
            except Exception:
                pass

            return self.failure_policy.handle(case, exc, partial_state, run_meta)

        finally:
            # Restore original checkpointer
            if cp_module is not None and original_cp is not None:
                cp_module._checkpointer = original_cp
            await conn.close()

    async def run_batch(
        self,
        cases: List[BenchmarkCase],
        on_result: Optional[Any] = None,
    ) -> List[EvalResult]:
        """
        Run multiple benchmark cases sequentially.

        Args:
            cases: List of benchmark cases.
            on_result: Optional callback(EvalResult) called after each case.

        Returns:
            List of EvalResult objects.
        """
        results: List[EvalResult] = []
        for i, case in enumerate(cases, 1):
            logger.info("Running case %d/%d: %s", i, len(cases), case.case_id)
            result = await self.run_case(case)
            results.append(result)
            if on_result:
                on_result(result)
        return results
