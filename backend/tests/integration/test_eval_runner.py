"""
Integration test for the EvalRunner.

Uses the same mocking pattern as the existing workflow integration tests:
in-memory AsyncSqliteSaver checkpointer + FakeChatModel + patched LLM factories.

Tests that the runner correctly:
1. Handles clarification interrupt → auto-resume
2. Handles reviewer interrupt → auto-approve
3. Captures report and findings in the result
"""

import pytest
import aiosqlite
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import Field

from evals.models import BenchmarkCase, RunStatus
from evals.runner import EvalRunner
from evals.scorers import SuccessScorer, FindingScorer

from app.models.schemas import (
    ResearchBrief,
    Finding,
    Citation,
    SubAgentSummary,
)
from app.agents.sub_agent import CitationExtractionOutput

import app.persistence.checkpointer as cp_module


class FakeChatModel(BaseChatModel):
    """Deterministic test double — same as workflow integration tests."""
    responses: list[Any] = Field(default_factory=lambda: ["default"])

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        res = self.responses.pop(0) if self.responses else "default"
        msg = res if isinstance(res, BaseMessage) else AIMessage(content=str(res))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, *a, **kw):
        return self

    def with_structured_output(self, schema, **kwargs):
        async def _ainvoke(input_data, config=None, **kw):
            res = self.responses.pop(0) if self.responses else None
            if isinstance(res, str) and hasattr(schema, "model_validate_json"):
                try:
                    return schema.model_validate_json(res)
                except Exception:
                    pass
            return res

        def _invoke(input_data, config=None, **kw):
            res = self.responses.pop(0) if self.responses else None
            if isinstance(res, str) and hasattr(schema, "model_validate_json"):
                try:
                    return schema.model_validate_json(res)
                except Exception:
                    pass
            return res

        r = RunnableLambda(_invoke)
        r.ainvoke = _ainvoke
        return r


@pytest.fixture
async def mock_db():
    """In-memory SQLite for checkpointer — same pattern as existing tests."""
    conn = await aiosqlite.connect(":memory:", isolation_level=None)
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()

    original = cp_module._checkpointer
    cp_module._checkpointer = checkpointer
    yield
    cp_module._checkpointer = original
    await conn.close()


@pytest.fixture
def mock_llms():
    chat = FakeChatModel()
    reasoner = FakeChatModel()
    reasoner_json = FakeChatModel()

    with patch("app.agents.scope_agent.get_deepseek_chat", return_value=chat), \
         patch("app.agents.sub_agent.get_deepseek_chat", return_value=chat), \
         patch("app.agents.supervisor_agent.get_deepseek_reasoner_json", return_value=reasoner_json), \
         patch("app.agents.report_agent.get_deepseek_reasoner", return_value=reasoner):
        yield {"chat": chat, "reasoner": reasoner, "reasoner_json": reasoner_json}


@pytest.fixture
def mock_tools():
    with patch("app.tools.academic.arxiv._search_arxiv_internal") as m_arxiv, \
         patch("app.tools.academic.semantic_scholar._search_semantic_scholar_sync") as m_ss, \
         patch("app.tools.academic.scopus._search_scopus_sync") as m_scopus:
        m_arxiv.return_value = []
        m_ss.return_value = []
        m_scopus.return_value = []
        yield {"arxiv": m_arxiv, "semantic_scholar": m_ss, "scopus": m_scopus}


@pytest.mark.asyncio
async def test_runner_handles_clarification_and_review(mock_db, mock_llms, mock_tools):
    """
    Full integration: vague query → clarification → brief → supervisor →
    report → reviewer auto-approve → SUCCESS.
    """
    # Scope round 1: incomplete → ask question
    mock_llms["chat"].responses = [
        '{"is_complete": false, "reasoning": "Need details", "missing_info": ["depth"]}',
        "What depth of coverage do you need?",
    ]

    case = BenchmarkCase(
        case_id="int_001",
        query="Tell me about AI",
        context="Focus on deep learning applications in NLP",
        expected_topics=["deep learning", "NLP"],
    )

    # After clarification resume: scope checks completion again
    mock_llms["chat"].responses += [
        '{"is_complete": true, "reasoning": "Got details", "missing_info": []}',
        '{"scope": "Deep learning in NLP", "sub_topics": ["transformers"], "constraints": {}, "deliverables": "Report", "format": "deep_research"}',
    ]

    # Supervisor gap analysis: no gaps, complete immediately.
    # The production graph now attempts an initial task decomposition before
    # gap analysis on the first supervisor pass, so bypass that branch here to
    # keep this test focused on the runner's interrupt/resume behavior.
    mock_llms["reasoner_json"].responses = [
        '{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Done"}'
    ]

    runner = EvalRunner(
        scorers=[SuccessScorer(), FindingScorer()],
        budget_overrides={"max_iterations": 3, "max_sub_agents": 2},
    )

    with patch(
        "app.agents.supervisor_agent._decompose_initial_tasks",
        new=AsyncMock(side_effect=RuntimeError("skip initial decomposition for this test")),
    ), patch(
        "app.agents.report_agent.generate_report",
        new=AsyncMock(return_value="# Final AI Report\n\nDeep learning in NLP..."),
    ):
        result = await runner.run_case(case)

    assert result.status == RunStatus.SUCCESS
    assert result.report_content is not None
    assert "Final AI Report" in result.report_content
    assert result.metadata.clarification_interrupts >= 1
    assert result.metadata.review_interrupts >= 1

    success_score = result.score_by_name("success")
    assert success_score == 1.0
