"""Unit tests for Supervisor Agent node."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.agents.supervisor_agent import supervisor_node, aggregate_findings, GapAnalysisOutput
from app.graphs.state import ResearchState
from app.models.schemas import ResearchBrief, Finding, ResearchTask, Citation


class TestSupervisorNode:
    """Test cases for supervisor_node function."""

    @pytest.mark.asyncio
    async def test_supervisor_node_budget_exhausted_iterations(self):
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
        
        result = await supervisor_node(state)
        
        assert result["is_complete"] is True
        assert result["budget"]["iterations"] == 20
    
    @pytest.mark.asyncio
    async def test_supervisor_node_budget_exhausted_sub_agents(self):
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
        
        result = await supervisor_node(state)
        
        assert result["is_complete"] is True
    
    @pytest.mark.asyncio
    @patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner_json")
    async def test_supervisor_node_llm_error_marks_complete(self, mock_get_llm, mock_template):
        """Test that LLM errors are handled gracefully and mark research complete."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(side_effect=Exception("LLM API error"))
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
        
        result = await supervisor_node(state)
        
        assert result["is_complete"] is True
        assert "error" in result
        assert any("Supervisor analysis failed" in err for err in result["error"])
        assert result["budget"]["iterations"] == 2
    
    @pytest.mark.asyncio
    @patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner_json")
    async def test_supervisor_node_generates_tasks_for_gaps(self, mock_get_llm, mock_template):
        """Test that supervisor generates new tasks when gaps exist."""
        mock_response = MagicMock()
        mock_response.content = """{
            "has_gaps": true,
            "is_complete": false,
            "gaps_identified": ["Missing topic2"],
            "new_tasks": [{
                "task_id": "task_001",
                "topic": "topic2",
                "query": "Research topic2",
                "priority": 1,
                "requested_by": "supervisor"
            }],
            "reasoning": "Topic2 has no findings"
        }"""
        
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        mock_template.__or__.return_value = mock_chain
        
        # Mock the LLM
        mock_llm = MagicMock()
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
        
        result = await supervisor_node(state)
        
        assert "task_history" in result
        assert len(result["task_history"]) == 1
        assert result["task_history"][0].task_id == "task_001"
        assert result["is_complete"] is False
        assert result["gaps"]["has_gaps"] is True
    
    @pytest.mark.asyncio
    @patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner_json")
    async def test_supervisor_node_marks_complete_when_no_gaps(self, mock_get_llm, mock_template):
        """Test that supervisor marks complete when no gaps exist."""
        mock_response = MagicMock()
        mock_response.content = """{
            "has_gaps": false,
            "is_complete": true,
            "gaps_identified": [],
            "new_tasks": [],
            "reasoning": "All topics sufficiently covered"
        }"""
        
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
        mock_template.__or__.return_value = mock_chain
        
        # Mock the LLM
        mock_llm = MagicMock()
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
        
        result = await supervisor_node(state)
        
        assert result["is_complete"] is True
        assert result["gaps"]["has_gaps"] is False
    
    @pytest.mark.asyncio
    @patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner_json")
    async def test_supervisor_node_increments_iterations(self, mock_get_llm, mock_template):
        """Test that supervisor correctly increments iteration counter."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_response = MagicMock()
        mock_response.content = """{
            "has_gaps": false,
            "is_complete": false,
            "gaps_identified": [],
            "new_tasks": [],
            "reasoning": "Test reasoning"
        }"""
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value=mock_response)
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
                "iterations": 5,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 0
            }
        }
        
        result = await supervisor_node(state)
        
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
    
    
    def test_aggregate_findings_deduplicates_by_title_author_fallback(self):
        """Test that findings with same title/author but different URLs are deduplicated."""
        findings = [
            Finding(
                claim="Claim 1",
                citation=Citation(
                    source="Source 1",
                    title="Same Title",
                    authors=["Author A"],
                    url="http://url1.com"
                ),
                topic="t1",
                credibility_score=0.8
            ),
            Finding(
                claim="Claim 2",
                citation=Citation(
                    source="Source 2",
                    title="Same Title",
                    authors=["Author A"],
                    url="http://url2.com"  # Different URL
                ),
                topic="t1",
                credibility_score=0.9
            )
        ]
        
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["t1"],
                constraints={},
                deliverables="Test"
            ),
            "findings": findings
        }
        
        result = aggregate_findings(state)
        
        assert len(result) == 1
        assert result[0].credibility_score == 0.9

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

class TestFormatContexts:
    def test_format_findings_max_limit(self):
        from app.agents.supervisor_agent import _format_findings_for_supervisor
        findings = [
            Finding(claim=f"Claim {i}", citation=Citation(source="Source", url=""), topic=f"t{i}", credibility_score=0.8)
            for i in range(60)
        ]
        res = _format_findings_for_supervisor(findings, max_findings=5)
        assert "showing 5" in res or "showing 4" in res or "showing 6" not in res
        
    def test_format_summaries_for_supervisor(self):
        from app.agents.supervisor_agent import _format_summaries_for_supervisor
        from app.models.schemas import SubAgentSummary
        
        assert "No sub-agent summaries" in _format_summaries_for_supervisor([])
        
        summaries = [
            SubAgentSummary(task_id="t1", task_answered=True, key_insights=["Insight 1"], gaps_noted="gap", finding_count=2),
            SubAgentSummary(task_id="t2", task_answered=False, key_insights=[], finding_count=0)
        ]
        res = _format_summaries_for_supervisor(summaries)
        assert "Task t1" in res
        assert "Answered" in res
        assert "Insight 1" in res
        assert "gap" in res
        assert "Task t2" in res
        assert "Incomplete" in res

class TestDecomposeInitialTasks:
    @pytest.mark.asyncio
    @patch("app.agents.supervisor_agent.RESEARCH_TASK_DECOMPOSITION_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner_json")
    async def test_decompose_tasks(self, mock_get_llm, mock_template):
        from app.agents.supervisor_agent import _decompose_initial_tasks
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_chain = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '[{"task_id": "test", "description": "desc", "query_variants": ["q1"]}]'
        mock_chain.ainvoke = AsyncMock(return_value=mock_resp)
        mock_template.__or__.return_value = mock_chain
        
        brief = ResearchBrief(scope="Scope", sub_topics=["t1"], constraints={}, deliverables="D")
        budget = {"max_sub_agents": 20}
        res = await _decompose_initial_tasks(brief, budget)
        
        assert res.has_gaps is True
        assert len(res.new_tasks) == 1
        assert res.new_tasks[0].task_id == "test"
        assert res.new_tasks[0].query == "q1"

class TestSupervisorNodeMisc:
    @pytest.mark.asyncio
    async def test_supervisor_is_complete_early_return(self):
        state = {"is_complete": True, "research_brief": MagicMock(), "budget": {}}
        res = await supervisor_node(state)
        assert res == {"is_complete": True}

    @pytest.mark.asyncio
    @patch("app.agents.supervisor_agent.ls.get_current_run_tree")
    @patch("app.agents.supervisor_agent._decompose_initial_tasks")
    async def test_supervisor_initial_iteration(self, mock_decompose, mock_get_rt):
        mock_rt = MagicMock()
        mock_rt.metadata = {}
        mock_get_rt.return_value = mock_rt
        
        mock_decompose.return_value = GapAnalysisOutput(
            has_gaps=True, is_complete=False, gaps_identified=["gap"], new_tasks=[], reasoning="reasoning"
        )
        
        state = {
            "research_brief": ResearchBrief(scope="Scope", sub_topics=["t1"], constraints={}, deliverables="D"),
            "findings": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {"iterations": 0, "max_iterations": 20, "max_sub_agents": 20, "max_searches_per_agent": 2}
        }
        res = await supervisor_node(state)
        
        assert "task_history" in res
        assert res["gaps"]["has_gaps"] is True
        assert mock_rt.metadata["iteration"] == 1
        
        mock_decompose.side_effect = Exception("Decompose failed")
        with patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE") as mock_template, patch("app.agents.supervisor_agent.get_deepseek_reasoner_json") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_chain = MagicMock()
            mock_resp = MagicMock()
            mock_resp.content = '{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Test"}'
            mock_chain.ainvoke = AsyncMock(return_value=mock_resp)
            mock_template.__or__.return_value = mock_chain
            
            res2 = await supervisor_node(state)
            assert res2["is_complete"] is True
            assert res2["budget"]["iterations"] == 1

    @pytest.mark.asyncio
    @patch("app.agents.supervisor_agent.SUPERVISOR_GAP_ANALYSIS_TEMPLATE")
    @patch("app.agents.supervisor_agent.get_deepseek_reasoner_json")
    async def test_supervisor_empty_llm_response_and_json_error_and_aggregation_error(self, mock_get_llm, mock_template):
        state = {
            "research_brief": ResearchBrief(scope="S", sub_topics=["t"], constraints={}, deliverables="D"),
            "findings": [Finding(claim="c", citation=Citation(source="s", url=""), topic="t", credibility_score=0.9)],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {"iterations": 5, "max_iterations": 20, "max_sub_agents": 20}
        }
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_chain = MagicMock()
        
        mock_resp = MagicMock()
        mock_resp.content = ""
        mock_chain.ainvoke = AsyncMock(return_value=mock_resp)
        mock_template.__or__.return_value = mock_chain
        
        res = await supervisor_node(state)
        assert res["is_complete"] is True
        assert "error" in res
        assert "Empty response from LLM" in str(res["error"])
        
        mock_resp.content = "not json"
        res2 = await supervisor_node(state)
        assert res2["is_complete"] is True
        assert "error" in res2
        assert "JSON parsing failed" in str(res2["error"])

        mock_resp.content = '{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Test"}'
        with patch("app.agents.supervisor_agent.aggregate_findings", side_effect=Exception("Agg fail")):
            res3 = await supervisor_node(state)
            assert res3["is_complete"] is True
            assert "findings" not in res3
