"""Unit tests for Supervisor Agent node.

Tests gap analysis, task generation, budget enforcement, and completion detection.
Follows backend-testing.md standards: happy path, edge cases, error handling.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.agents.supervisor_agent import supervisor_node, aggregate_findings, GapAnalysisOutput
from app.graphs.state import ResearchState
from app.models.schemas import ResearchBrief, Finding, ResearchTask, Citation


class TestSupervisorNode:
    """Test cases for supervisor_node function."""

    def test_supervisor_node_budget_exhausted_iterations(self):
        """Test that supervisor completes when max_iterations reached."""
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1", "topic2"],
                constraints={},
                deliverables="Test"
            ),
            "findings": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {
                "iterations": 19,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 0
            }
        }
        
        result = supervisor_node(state)
        
        assert result["is_complete"] is True
        assert result["budget"]["iterations"] == 20
    
    def test_supervisor_node_budget_exhausted_sub_agents(self):
        """Test that supervisor completes when max_sub_agents reached."""
        findings = [
            Finding(
                claim=f"Claim {i}",
                citation=Citation(source="Source", url="http://example.com"),
                topic="topic1",
                credibility_score=0.8
            )
            for i in range(20)
        ]
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": findings,
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {
                "iterations": 5,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 10
            }
        }
        
        result = supervisor_node(state)
        
        assert result["is_complete"] is True
    
    @patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner")
    def test_supervisor_node_llm_error_marks_complete(self, mock_get_llm, mock_template):
        """Test that LLM errors are handled gracefully and mark research complete."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock the chain to raise exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM API error")
        mock_template.__or__.return_value = mock_chain
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {
                "iterations": 1,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 0
            }
        }
        
        result = supervisor_node(state)
        
        assert result["is_complete"] is True
        assert "error" in result
        assert "Supervisor analysis failed" in result["error"]
        assert result["budget"]["iterations"] == 2
    
    @patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner")
    def test_supervisor_node_generates_tasks_for_gaps(self, mock_get_llm, mock_template):
        """Test that supervisor generates new tasks when gaps exist."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        
        gap_output = GapAnalysisOutput(
            has_gaps=True,
            is_complete=False,
            gaps_identified=["Missing topic2"],
            new_tasks=[
                ResearchTask(
                    task_id="task_001",
                    topic="topic2",
                    query="Research topic2",
                    priority=1
                )
            ],
            reasoning="Topic2 has no findings"
        )
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = gap_output
        mock_template.__or__ = MagicMock(return_value=mock_chain)
        
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1", "topic2"],
                constraints={},
                deliverables="Test"
            ),
            "findings": [
                Finding(
                    claim="Claim about topic1",
                    citation=Citation(source="Source", url="http://example.com"),
                    topic="topic1",
                    credibility_score=0.8
                )
            ],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {
                "iterations": 1,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 2
            }
        }
        
        result = supervisor_node(state)
        
        assert "task_history" in result
        assert len(result["task_history"]) == 1
        assert result["task_history"][0].task_id == "task_001"
        assert result["is_complete"] is False
        assert result["gaps"]["has_gaps"] is True
    
    @patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner")
    def test_supervisor_node_marks_complete_when_no_gaps(self, mock_get_llm, mock_template):
        """Test that supervisor marks complete when no gaps exist."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        
        gap_output = GapAnalysisOutput(
            has_gaps=False,
            is_complete=True,
            gaps_identified=[],
            new_tasks=[],
            reasoning="All topics sufficiently covered"
        )
        
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = gap_output
        mock_template.__or__ = MagicMock(return_value=mock_chain)
        
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": [
                Finding(
                    claim="Claim 1",
                    citation=Citation(source="Source", url="http://example.com"),
                    topic="topic1",
                    credibility_score=0.9
                ),
                Finding(
                    claim="Claim 2",
                    citation=Citation(source="Source2", url="http://example2.com"),
                    topic="topic1",
                    credibility_score=0.8
                )
            ],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {
                "iterations": 2,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 4
            }
        }
        
        result = supervisor_node(state)
        
        assert result["is_complete"] is True
        assert result["gaps"]["has_gaps"] is False
    
    def test_supervisor_node_increments_iterations(self):
        """Test that supervisor correctly increments iteration counter."""
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {
                "iterations": 5,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 0
            }
        }
        
        result = supervisor_node(state)
        
        assert result["budget"]["iterations"] == 6


class TestAggregateFindings:
    """Test cases for aggregate_findings function."""
    
    def test_aggregate_findings_filters_low_credibility(self):
        """Test that low credibility findings are filtered out."""
        findings = [
            Finding(
                claim="High credibility claim",
                citation=Citation(source="Nature", url="http://nature.com/1"),
                topic="topic1",
                credibility_score=0.9
            ),
            Finding(
                claim="Low credibility claim",
                citation=Citation(source="Blog", url="http://blog.com/1"),
                topic="topic1",
                credibility_score=0.3
            ),
            Finding(
                claim="Medium credibility claim",
                citation=Citation(source="ArXiv", url="http://arxiv.org/1"),
                topic="topic1",
                credibility_score=0.7
            )
        ]
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": findings
        }
        
        result = aggregate_findings(state)
        
        assert len(result) == 2
        assert all(f.credibility_score >= 0.5 for f in result)
    
    def test_aggregate_findings_deduplicates_by_doi(self):
        """Test that findings with same DOI are deduplicated."""
        findings = [
            Finding(
                claim="First claim",
                citation=Citation(
                    source="Nature",
                    url="http://nature.com/1",
                    doi="10.1038/test123"
                ),
                topic="topic1",
                credibility_score=0.8
            ),
            Finding(
                claim="Same paper different claim",
                citation=Citation(
                    source="Nature",
                    url="http://nature.com/2",
                    doi="10.1038/test123"
                ),
                topic="topic1",
                credibility_score=0.9
            )
        ]
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": findings
        }
        
        result = aggregate_findings(state)
        
        assert len(result) == 1
        assert result[0].credibility_score == 0.9
    
    def test_aggregate_findings_deduplicates_by_url(self):
        """Test that findings with same URL are deduplicated."""
        findings = [
            Finding(
                claim="First claim",
                citation=Citation(
                    source="Blog",
                    url="http://example.com/article"
                ),
                topic="topic1",
                credibility_score=0.6
            ),
            Finding(
                claim="Same URL different claim",
                citation=Citation(
                    source="Blog",
                    url="http://example.com/article"
                ),
                topic="topic1",
                credibility_score=0.7
            )
        ]
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": findings
        }
        
        result = aggregate_findings(state)
        
        assert len(result) == 1
        assert result[0].credibility_score == 0.7
    
    
    def test_aggregate_findings_sorts_by_credibility(self):
        """Test that findings are sorted by credibility score descending."""
        findings = [
            Finding(
                claim="Low score",
                citation=Citation(source="S1", url="http://s1.com"),
                topic="topic1",
                credibility_score=0.6
            ),
            Finding(
                claim="High score",
                citation=Citation(source="S2", url="http://s2.com"),
                topic="topic1",
                credibility_score=0.95
            ),
            Finding(
                claim="Medium score",
                citation=Citation(source="S3", url="http://s3.com"),
                topic="topic1",
                credibility_score=0.75
            )
        ]
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": findings
        }
        
        result = aggregate_findings(state)
        
        assert len(result) == 3
        assert result[0].credibility_score == 0.95
        assert result[1].credibility_score == 0.75
        assert result[2].credibility_score == 0.6
    
    def test_aggregate_findings_empty_list_returns_empty(self):
        """Test that empty findings list returns empty result."""
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": []
        }
        
        result = aggregate_findings(state)
        
        assert result == []
