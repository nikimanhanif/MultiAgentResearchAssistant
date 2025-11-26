"""Evaluation tests for Report Agent using DeepEval metrics.

These tests use OpenAI gpt-5-nano for evaluation (independent of DeepSeek research LLM).
Requires OPENAI_API_KEY to run.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from unittest.mock import AsyncMock, patch

from app.agents.report_agent import generate_report
from app.models.schemas import ResearchBrief, Finding, Citation, ReportFormat, SourceType


@pytest.fixture
def evaluation_brief():
    return ResearchBrief(
        scope="Impact of AI on Healthcare",
        sub_topics=["Diagnostics", "Treatment Planning", "Patient Care"],
        constraints={"time_period": "2020-2024", "focus": "clinical applications"},
        deliverables="Literature review",
        format=ReportFormat.LITERATURE_REVIEW
    )


@pytest.fixture
def evaluation_findings():
    return [
        Finding(
            claim="AI systems achieve 95% accuracy in diagnosing certain cancers from medical imaging",
            citation=Citation(
                source="Journal of Medical AI",
                url="https://example.com/jmai/article1",
                title="Deep Learning for Cancer Detection",
                authors=["Zhang, L.", "Kumar, S.", "Chen, M."],
                year=2023,
                credibility_score=0.92,
                source_type=SourceType.PEER_REVIEWED,
                doi="10.1234/jmai.2023.001"
            ),
            topic="Diagnostics",
            credibility_score=0.92
        ),
        Finding(
            claim="Machine learning models can predict patient outcomes with 85% accuracy",
            citation=Citation(
                source="Healthcare Technology Review",
                url="https://example.com/htr/article2",
                title="Predictive Analytics in Healthcare",
                authors=["Smith, J.", "Johnson, R."],
                year=2022,
                credibility_score=0.88,
                source_type=SourceType.PEER_REVIEWED,
                doi="10.1234/htr.2022.045"
            ),
            topic="Treatment Planning",
            credibility_score=0.88
        ),
        Finding(
            claim="AI chatbots improve patient engagement scores by 40%",
            citation=Citation(
                source="Digital Health Blog",
                url="https://example.com/blog/ai-chatbots",
                title="AI Chatbots in Healthcare",
                authors=None,
                year=2023,
                credibility_score=0.45,
                source_type=SourceType.BLOG
            ),
            topic="Patient Care",
            credibility_score=0.45
        ),
        Finding(
            claim="Automated treatment planning systems reduce planning time by 60%",
            citation=Citation(
                source="Clinical Oncology Journal",
                url="https://example.com/coj/article3",
                title="Automation in Radiation Therapy",
                authors=["Williams, A.", "Davis, K."],
                year=2024,
                credibility_score=0.90,
                source_type=SourceType.PEER_REVIEWED,
                doi="10.1234/coj.2024.012"
            ),
            topic="Treatment Planning",
            credibility_score=0.90
        )
    ]


@pytest.fixture
def mock_llm_response():
    return """# AI in Healthcare: A Literature Review

## Introduction

Artificial intelligence has transformed healthcare across multiple domains. This review examines recent advances in AI applications for diagnostics, treatment planning, and patient care based on recent research from 2020-2024.

## Diagnostics

Recent studies demonstrate significant progress in AI-powered diagnostic systems. AI systems achieve 95% accuracy in diagnosing certain cancers from medical imaging [0]. These deep learning models have shown particular promise in radiology and pathology applications.

## Treatment Planning

Machine learning has revolutionized treatment planning workflows. Machine learning models can predict patient outcomes with 85% accuracy [1], enabling more personalized treatment strategies. Furthermore, automated treatment planning systems reduce planning time by 60% [3], significantly improving clinical efficiency.

## Patient Care

AI applications in patient care show mixed results. While AI chatbots improve patient engagement scores by 40% [2], this finding comes from a lower-credibility source and should be interpreted cautiously.

## Conclusion

AI demonstrates substantial potential across healthcare domains, with strongest evidence in diagnostics and treatment planning. Future research should focus on validation and integration into clinical workflows.

## References

[0] Zhang, L., Kumar, S., Chen, M. (2023). "Deep Learning for Cancer Detection". Journal of Medical AI. DOI: 10.1234/jmai.2023.001
    https://example.com/jmai/article1 (Credibility: 0.92 - High)

[1] Smith, J., Johnson, R. (2022). "Predictive Analytics in Healthcare". Healthcare Technology Review. DOI: 10.1234/htr.2022.045
    https://example.com/htr/article2 (Credibility: 0.88 - High)

[2] Unknown Author. "AI Chatbots in Healthcare". Digital Health Blog. https://example.com/blog/ai-chatbots
    ⚠️ (Credibility: 0.45 - Low Quality Source)

[3] Williams, A., Davis, K. (2024). "Automation in Radiation Therapy". Clinical Oncology Journal. DOI: 10.1234/coj.2024.012
    https://example.com/coj/article3 (Credibility: 0.90 - High)
"""


@pytest.mark.evaluation
class TestReportAgentEvaluation:
    """Evaluation tests for Report Agent using DeepEval metrics."""

    @pytest.mark.asyncio
    async def test_report_faithfulness_metric(
        self, evaluation_brief, evaluation_findings, mock_llm_response
    ):
        """Test that report does not hallucinate beyond provided findings."""
        with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=AsyncMock(content=mock_llm_response))
            mock_llm_factory.return_value = mock_llm

            actual_output = await generate_report(evaluation_brief, evaluation_findings)

            retrieval_context = [f.claim for f in evaluation_findings]

            test_case = LLMTestCase(
                input=evaluation_brief.scope,
                actual_output=actual_output,
                retrieval_context=retrieval_context
            )

            faithfulness_metric = FaithfulnessMetric(
                threshold=0.9,
                model="gpt-5-nano",
                include_reason=True
            )

            assert_test(test_case, [faithfulness_metric])

    @pytest.mark.asyncio
    async def test_report_format_compliance_metric(
        self, evaluation_brief, evaluation_findings, mock_llm_response
    ):
        """Test that report follows requested format structure."""
        with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=AsyncMock(content=mock_llm_response))
            mock_llm_factory.return_value = mock_llm

            actual_output = await generate_report(evaluation_brief, evaluation_findings)

            test_case = LLMTestCase(
                input=evaluation_brief.scope,
                actual_output=actual_output
            )

            format_compliance_metric = GEval(
                name="Format Compliance",
                criteria="Determine whether the actual output follows the literature review format structure with Introduction, thematic sections, conclusion, and bibliography.",
                evaluation_steps=[
                    "Check if the output has a clear introduction section",
                    "Verify that findings are organized by thematic sections (sub-topics)",
                    "Confirm there is a conclusion section",
                    "Verify there is a references/bibliography section with proper citations",
                    "Check if citations use numbered format [0], [1], etc."
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=0.8,
                model="gpt-5-nano"
            )

            assert_test(test_case, [format_compliance_metric])

    @pytest.mark.asyncio
    async def test_report_citation_quality_metric(
        self, evaluation_brief, evaluation_findings, mock_llm_response
    ):
        """Test that citations are properly formatted and include credibility indicators."""
        with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
            mock_llm = AsyncMock()
            mock_llm.ainvoke = AsyncMock(return_value=AsyncMock(content=mock_llm_response))
            mock_llm_factory.return_value = mock_llm

            actual_output = await generate_report(evaluation_brief, evaluation_findings)

            test_case = LLMTestCase(
                input=evaluation_brief.scope,
                actual_output=actual_output
            )

            citation_quality_metric = GEval(
                name="Citation Quality",
                criteria="Determine whether citations are properly formatted with credibility indicators and warnings for low-quality sources.",
                evaluation_steps=[
                    "Check if in-text citations use numbered format [0], [1], etc.",
                    "Verify that bibliography includes all citation details (authors, title, source, DOI, URL)",
                    "Confirm that credibility scores are included in bibliography",
                    "Check if low-credibility sources (< 0.5) have warning indicators",
                    "Verify that all cited sources appear in the bibliography"
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=0.8,
                model="gpt-5-nano"
            )

            assert_test(test_case, [citation_quality_metric])
