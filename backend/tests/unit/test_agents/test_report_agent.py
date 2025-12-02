"""Unit tests for Report Agent."""

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.runnables import Runnable

from app.agents.report_agent import (
    generate_report,
    _build_report_generation_chain,
    _get_format_instructions,
    _generate_no_findings_report,
    report_agent_node,
)
from app.models.schemas import ResearchBrief, Finding, Citation, ReportFormat, SourceType
from app.graphs.state import ResearchState



class TestBuildReportGenerationChain:
    """Tests for _build_report_generation_chain builder function."""

    def test_build_report_generation_chain_returns_runnable(self):
        with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
            with patch('app.agents.report_agent.get_report_generation_prompt') as mock_prompt:
                mock_prompt.return_value = AsyncMock()
                mock_llm_factory.return_value = AsyncMock()

                chain = _build_report_generation_chain()

                assert chain is not None
                mock_prompt.assert_called_once()
                mock_llm_factory.assert_called_once_with(temperature=0.5)

    def test_build_report_generation_chain_llm_init_failure_raises(self):
        with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
            mock_llm_factory.side_effect = ValueError("API key not configured")

            with pytest.raises(ValueError, match="API key not configured"):
                _build_report_generation_chain()


class TestGenerateReport:
    """Tests for generate_report function."""

    @pytest.fixture
    def sample_brief(self):
        return ResearchBrief(
            scope="Machine Learning Applications",
            sub_topics=["Computer Vision", "NLP"],
            constraints={"time_period": "2020-2024"},
            deliverables="Comprehensive review",
            format=ReportFormat.SUMMARY
        )

    @pytest.fixture
    def sample_findings(self):
        return [
            Finding(
                claim="Deep learning models excel at image classification",
                citation=Citation(
                    source="Nature AI",
                    url="https://nature.com/article1",
                    title="Deep Learning Advances",
                    authors=["Smith, J.", "Doe, A."],
                    year=2023,
                    credibility_score=0.95,
                    source_type=SourceType.PEER_REVIEWED,
                    doi="10.1038/example1"
                ),
                topic="Computer Vision",
                credibility_score=0.95
            ),
            Finding(
                claim="Transformers revolutionized NLP",
                citation=Citation(
                    source="Blog Post",
                    url="https://example.com/blog",
                    title="Transformers Explained",
                    authors=None,
                    year=2022,
                    credibility_score=0.35,
                    source_type=SourceType.BLOG
                ),
                topic="NLP",
                credibility_score=0.35
            )
        ]

    @pytest.mark.asyncio
    async def test_generate_report_valid_findings_returns_markdown(
        self, sample_brief, sample_findings
    ):
        with patch('app.agents.report_agent.get_report_generation_prompt') as mock_prompt:
            with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
                mock_content = "# Machine Learning Report\\n\\n## Introduction\\n\\nTest report content [0].\\n\\n## References\\n\\n[0] Smith, J., Doe, A. (2023)"
                
                mock_chain = AsyncMock()
                mock_chain.ainvoke = AsyncMock(return_value=type('Response', (), {'content': mock_content})())
                
                mock_prompt.return_value.__or__ = lambda self, other: mock_chain
                mock_llm_factory.return_value = AsyncMock()

                result = await generate_report(sample_brief, sample_findings)

                assert isinstance(result, str)
                assert len(result) > 0
                assert "Machine Learning" in result or "Introduction" in result

    @pytest.mark.asyncio
    async def test_generate_report_empty_findings_returns_minimal_report(self, sample_brief):
        result = await generate_report(sample_brief, [])

        assert isinstance(result, str)
        assert "Machine Learning Applications" in result
        assert "No research findings" in result
        assert "Computer Vision" in result
        assert "NLP" in result

    @pytest.mark.asyncio
    async def test_generate_report_includes_citations_in_correct_format(
        self, sample_brief, sample_findings
    ):
        with patch('app.agents.report_agent.get_report_generation_prompt') as mock_prompt:
            with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
                mock_content = "Report with citations [0] and [1]\\n\\n## References\\n\\n[0] First citation\\n[1] Second citation"
                
                mock_chain = AsyncMock()
                mock_chain.ainvoke = AsyncMock(return_value=type('Response', (), {'content': mock_content})())
                
                mock_prompt.return_value.__or__ = lambda self, other: mock_chain
                mock_llm_factory.return_value = AsyncMock()

                result = await generate_report(sample_brief, sample_findings)

                assert "[0]" in result or "[1]" in result or "References" in result

    @pytest.mark.asyncio
    async def test_generate_report_none_brief_raises_error(self, sample_findings):
        with pytest.raises(ValueError, match="Research brief cannot be None"):
            await generate_report(None, sample_findings)

    @pytest.mark.asyncio
    async def test_generate_report_empty_scope_raises_error(self, sample_findings):
        brief = ResearchBrief(
            scope="",
            sub_topics=["Topic1"],
            constraints={},
            deliverables="Test"
        )
        with pytest.raises(ValueError, match="Research brief must have a scope"):
            await generate_report(brief, sample_findings)

    @pytest.mark.asyncio
    async def test_generate_report_none_findings_raises_error(self, sample_brief):
        with pytest.raises(ValueError, match="Findings list cannot be None"):
            await generate_report(sample_brief, None)

    @pytest.mark.asyncio
    async def test_generate_report_llm_init_failure_raises_error(
        self, sample_brief, sample_findings
    ):
        with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
            mock_llm_factory.side_effect = ValueError("API key not configured")

            with pytest.raises(Exception, match="Failed to initialize LLM"):
                await generate_report(sample_brief, sample_findings)

    @pytest.mark.asyncio
    async def test_generate_report_llm_generation_failure_raises_error(
        self, sample_brief, sample_findings
    ):
        with patch('app.agents.report_agent.get_report_generation_prompt') as mock_prompt:
            with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
                mock_chain = AsyncMock()
                mock_chain.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))
                
                mock_prompt.return_value.__or__ = lambda self, other: mock_chain
                mock_llm_factory.return_value = AsyncMock()

                with pytest.raises(Exception, match="Failed to generate report"):
                    await generate_report(sample_brief, sample_findings)

    @pytest.mark.asyncio
    async def test_generate_report_different_formats(self, sample_findings):
        formats = [
            ReportFormat.SUMMARY,
            ReportFormat.COMPARISON,
            ReportFormat.LITERATURE_REVIEW,
            ReportFormat.GAP_ANALYSIS,
            ReportFormat.FACT_VALIDATION,
            ReportFormat.RANKING,
            ReportFormat.OTHER
        ]

        for fmt in formats:
            brief = ResearchBrief(
                scope="Test Scope",
                sub_topics=["Topic1"],
                constraints={},
                deliverables="Test",
                format=fmt
            )

            with patch('app.agents.report_agent.get_report_generation_prompt') as mock_prompt:
                with patch('app.agents.report_agent.get_deepseek_reasoner') as mock_llm_factory:
                    mock_content = f"Report for {fmt.value}"
                    
                    mock_chain = AsyncMock()
                    mock_chain.ainvoke = AsyncMock(return_value=type('Response', (), {'content': mock_content})())
                    
                    mock_prompt.return_value.__or__ = lambda self, other: mock_chain
                    mock_llm_factory.return_value = AsyncMock()

                    result = await generate_report(brief, sample_findings)
                    assert isinstance(result, str)


class TestGetFormatInstructions:
    """Tests for _get_format_instructions helper function."""

    def test_get_format_instructions_summary(self):
        result = _get_format_instructions(ReportFormat.SUMMARY)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Summary" in result or "summary" in result

    def test_get_format_instructions_comparison(self):
        result = _get_format_instructions(ReportFormat.COMPARISON)
        assert isinstance(result, str)
        assert "comparison" in result.lower()

    def test_get_format_instructions_other(self):
        result = _get_format_instructions(ReportFormat.OTHER)
        assert isinstance(result, str)
        assert "general" in result.lower() or "Introduction" in result


class TestGenerateNoFindingsReport:
    """Tests for _generate_no_findings_report helper function."""

    def test_generate_no_findings_report_contains_scope(self):
        brief = ResearchBrief(
            scope="Test Research Scope",
            sub_topics=["Topic1", "Topic2"],
            constraints={},
            deliverables="Test"
        )
        result = _generate_no_findings_report(brief)

        assert "Test Research Scope" in result
        assert "Topic1" in result
        assert "Topic2" in result
        assert "No research findings" in result

    def test_generate_no_findings_report_is_valid_markdown(self):
        brief = ResearchBrief(
            scope="Markdown Test",
            sub_topics=["A", "B"],
            constraints={},
            deliverables="Test"
        )
        result = _generate_no_findings_report(brief)

        assert result.startswith("#")
        assert "##" in result


class TestReportAgentNode:
    """Tests for report_agent_node LangGraph integration."""

    @pytest.mark.asyncio
    async def test_report_agent_node_generates_report_successfully(self):
        """Test that report_agent_node generates report and updates state."""
        # Arrange
        brief = ResearchBrief(scope="Test", sub_topics=[], constraints={}, deliverables="")
        state = ResearchState(
            research_brief=brief,
            findings=[],
            reviewer_feedback=None
        )
        
        with patch("app.agents.report_agent.generate_report") as mock_generate:
            mock_generate.return_value = "# Final Report"
            
            # Act
            result = await report_agent_node(state)
            
            # Assert
            assert "report_content" in result
            assert result["report_content"] == "# Final Report"
            assert result["reviewer_feedback"] is None
            mock_generate.assert_called_once_with(brief, [], None)

    @pytest.mark.asyncio
    async def test_report_agent_node_handles_missing_brief(self):
        """Test that report_agent_node handles missing research brief."""
        # Arrange
        state = ResearchState(findings=[])
        
        # Act
        result = await report_agent_node(state)
        
        # Assert
        assert "error" in result
        assert "Missing research brief" in result["error"]
        assert "report_content" in result
        assert "Error" in result["report_content"]

    @pytest.mark.asyncio
    async def test_report_agent_node_passes_reviewer_feedback(self):
        """Test that report_agent_node passes reviewer feedback to generator."""
        # Arrange
        brief = ResearchBrief(scope="Test", sub_topics=[], constraints={}, deliverables="")
        state = ResearchState(
            research_brief=brief,
            findings=[],
            reviewer_feedback="Make it shorter"
        )
        
        with patch("app.agents.report_agent.generate_report") as mock_generate:
            mock_generate.return_value = "# Updated Report"
            
            # Act
            result = await report_agent_node(state)
            
            # Assert
            assert result["report_content"] == "# Updated Report"
            mock_generate.assert_called_once_with(brief, [], "Make it shorter")

    @pytest.mark.asyncio
    async def test_report_agent_node_handles_generation_exception(self):
        """Test that report_agent_node handles exceptions during generation."""
        # Arrange
        brief = ResearchBrief(scope="Test", sub_topics=[], constraints={}, deliverables="")
        state = ResearchState(research_brief=brief)
        
        with patch("app.agents.report_agent.generate_report") as mock_generate:
            mock_generate.side_effect = Exception("Generation failed")
            
            # Act
            result = await report_agent_node(state)
            
            # Assert
            assert "error" in result
            assert "Generation failed" in result["error"]
            assert "report_content" in result
            assert "Error generating report" in result["report_content"]
