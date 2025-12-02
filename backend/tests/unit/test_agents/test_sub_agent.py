"""Unit tests for Sub-Agent node.

Tests research execution, citation extraction, delegation parsing, and budget enforcement.
Follows backend-testing.md standards: happy path, edge cases, error handling.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.models.schemas import ResearchTask, Finding, Citation, SourceType
from app.agents.sub_agent import (
    sub_agent_node,
    _parse_delegation_request,
    _extract_citations,
    CitationExtractionOutput
)
from langchain_core.messages import ToolMessage, AIMessage
from app.graphs.state import SubAgentState
from app.models.schemas import ResearchBrief


class TestParseDelegationRequest:
    """Test cases for _parse_delegation_request function."""
    
    def test_parse_delegation_request_valid_format(self):
        """Test parsing valid delegation request."""
        output = "DELEGATION_REQUEST: topic='quantum error correction', reason='Need deeper analysis of stabilizer codes'"
        
        result = _parse_delegation_request(output)
        
        assert result is not None
        assert result.topic == "quantum error correction"
        assert "stabilizer codes" in result.query
        assert result.priority == 2
        assert result.requested_by == "sub_agent"
    
    def test_parse_delegation_request_no_match(self):
        """Test parsing when no delegation request present."""
        output = "This is a normal research output with findings."
        
        result = _parse_delegation_request(output)
        
        assert result is None
    
    def test_parse_delegation_request_malformed_returns_none(self):
        """Test that malformed delegation request returns None."""
        output = "DELEGATION_REQUEST: topic="
        
        result = _parse_delegation_request(output)
        
        assert result is None
    
    def test_parse_delegation_request_missing_reason_returns_none(self):
        """Test that incomplete delegation request returns None."""
        output = "DELEGATION_REQUEST: topic='quantum computing'"
        
        result = _parse_delegation_request(output)
        
        assert result is None


class TestSubAgentNode:
    """Test cases for sub_agent_node function."""
    
    @pytest.mark.asyncio
    async def test_sub_agent_node_skips_completed_task(self):
        """Test that sub-agent skips already completed tasks."""
        state: SubAgentState = {
            "task": ResearchTask(
                task_id="task_001",
                topic="test topic",
                query="test query",
                priority=1
            ),
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["test topic"],
                constraints={},
                deliverables="Test"
            ),
            "completed_tasks": ["task_001"],
            "failed_tasks": [],
            "budget": {
                "iterations": 1,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 0
            }
        }
        
        result = await sub_agent_node(state)
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_sub_agent_node_skips_failed_task(self):
        """Test that sub-agent skips already failed tasks."""
        state: SubAgentState = {
            "task": ResearchTask(
                task_id="task_002",
                topic="test topic",
                query="test query",
                priority=1
            ),
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["test topic"],
                constraints={},
                deliverables="Test"
            ),
            "completed_tasks": [],
            "failed_tasks": ["task_002"],
            "budget": {
                "iterations": 1,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 0
            }
        }
        
        result = await sub_agent_node(state)
        
        assert result == {}
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.get_research_tools")
    async def test_sub_agent_node_no_tools_available_fails(self, mock_get_tools):
        """Test that sub-agent fails gracefully when no tools available."""
        # Mock context manager returning empty list
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = []
        mock_ctx.__aexit__.return_value = None
        mock_get_tools.return_value = mock_ctx
        
        state: SubAgentState = {
            "task": ResearchTask(
                task_id="task_003",
                topic="test topic",
                query="test query",
                priority=1
            ),
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["test topic"],
                constraints={},
                deliverables="Test"
            ),
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
        
        result = await sub_agent_node(state)
        
        assert "failed_tasks" in result
        assert "task_003" in result["failed_tasks"]
        assert "error" in result
        assert "No tools available" in result["error"][0]
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent._extract_citations")
    @patch("app.agents.sub_agent.create_react_agent")
    @patch("app.agents.sub_agent.get_research_tools")
    @patch("app.agents.sub_agent.get_deepseek_chat")
    async def test_sub_agent_node_successful_execution_returns_findings(
        self, mock_get_llm, mock_get_tools, mock_create_agent, mock_extract
    ):
        """Test happy path: sub-agent successfully executes and returns findings."""
        # Mock tools
        mock_tool = MagicMock()
        mock_tool.name = "tavily_search"
        
        # Mock context manager
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = [mock_tool]
        mock_ctx.__aexit__.return_value = None
        mock_get_tools.return_value = mock_ctx
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock agent execution
        # Mock agent execution
        mock_agent = AsyncMock()
        
        # AI Message triggering tool
        mock_ai_msg = MagicMock(spec=AIMessage)
        mock_ai_msg.tool_calls = [{"name": "tavily_search", "args": {}, "id": "call_1"}]
        
        # Tool Message with result
        mock_tool_msg = MagicMock(spec=ToolMessage)
        mock_tool_msg.name = "tavily_search"
        mock_tool_msg.content = "Search results"
        mock_tool_msg.tool_call_id = "call_1"
        
        mock_final_msg = MagicMock()
        mock_final_msg.content = "Found information about quantum computing"
        
        mock_agent.ainvoke.return_value = {
            "messages": [mock_ai_msg, mock_tool_msg, mock_final_msg]
        }
        mock_create_agent.return_value = mock_agent
        
        # Mock citation extraction
        finding = Finding(
            claim="Quantum computing is advancing",
            citation=Citation(source="Nature", url="http://nature.com/1"),
            topic="quantum computing",
            credibility_score=0.9
        )
        mock_extract.return_value = [finding]
        
        state: SubAgentState = {
            "task": ResearchTask(
                task_id="task_004",
                topic="quantum computing",
                query="Research quantum computing advancements",
                priority=1
            ),
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["quantum computing"],
                constraints={},
                deliverables="Test"
            ),
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
        
        result = await sub_agent_node(state)
        
        assert "findings" in result
        assert len(result["findings"]) == 1
        assert result["findings"][0].claim == "Quantum computing is advancing"
        assert "completed_tasks" in result
        assert "task_004" in result["completed_tasks"]
        assert "budget" in result
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent._extract_citations")
    @patch("app.agents.sub_agent._parse_delegation_request")
    @patch("app.agents.sub_agent.create_react_agent")
    @patch("app.agents.sub_agent.get_research_tools")
    @patch("app.agents.sub_agent.get_deepseek_chat")
    async def test_sub_agent_node_delegation_adds_task_to_history(
        self, mock_get_llm, mock_get_tools, mock_create_agent, mock_parse_delegation, mock_extract
    ):
        """Test that delegation requests are parsed and added to task_history."""
        # Mock tools and LLM
        mock_tool = MagicMock()
        mock_tool.name = "tavily_search"
        
        # Mock context manager
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = [mock_tool]
        mock_ctx.__aexit__.return_value = None
        mock_get_tools.return_value = mock_ctx
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock agent with delegation in output AND tool results
        mock_agent = AsyncMock()
        
        mock_ai_msg = MagicMock(spec=AIMessage)
        mock_ai_msg.tool_calls = [{"name": "tavily_search", "args": {}, "id": "call_1"}]
        
        mock_tool_msg = MagicMock(spec=ToolMessage)
        mock_tool_msg.name = "tavily_search"
        mock_tool_msg.content = "Search results"
        mock_tool_msg.tool_call_id = "call_1"
        
        mock_final_msg = MagicMock()
        mock_final_msg.content = "DELEGATION_REQUEST: topic='subtopic', reason='deeper research needed'"
        
        mock_agent.ainvoke.return_value = {
            "messages": [mock_ai_msg, mock_tool_msg, mock_final_msg]
        }
        mock_create_agent.return_value = mock_agent
        
        # Mock delegation parsing
        delegated_task = ResearchTask(
            task_id="delegated_subtopic",
            topic="subtopic",
            query="subtopic - deeper research needed",
            priority=2,
            requested_by="sub_agent"
        )
        mock_parse_delegation.return_value = delegated_task
        
        # Mock citation extraction
        mock_extract.return_value = []
        
        state: SubAgentState = {
            "task": ResearchTask(
                task_id="task_005",
                topic="main topic",
                query="Research main topic",
                priority=1
            ),
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["main topic"],
                constraints={},
                deliverables="Test"
            ),
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
        
        result = await sub_agent_node(state)
        
        assert "task_history" in result
        assert len(result["task_history"]) == 1
        assert result["task_history"][0].task_id == "delegated_subtopic"
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.get_research_tools")
    @patch("app.agents.sub_agent.get_deepseek_chat")
    @patch("app.agents.sub_agent.create_react_agent")
    async def test_sub_agent_node_agent_failure_marks_failed(
        self, mock_create_agent, mock_get_llm, mock_get_tools
    ):
        """Test that agent execution failures are handled and task marked as failed."""
        # Mock tools and LLM
        mock_tool = MagicMock()
        mock_tool.name = "tavily_search"
        
        # Mock context manager
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = [mock_tool]
        mock_ctx.__aexit__.return_value = None
        mock_get_tools.return_value = mock_ctx
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock agent failure
        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = Exception("Agent execution failed")
        mock_create_agent.return_value = mock_agent
        
        state: SubAgentState = {
            "task": ResearchTask(
                task_id="task_006",
                topic="test topic",
                query="test query",
                priority=1
            ),
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["test topic"],
                constraints={},
                deliverables="Test"
            ),
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {
                "iterations": 1,
                "max_iterations": 20,
                "max_sub_agents": 20,
                "max_searches_per_agent": 2,
                "total_searches": 5
            }
        }
        
        result = await sub_agent_node(state)
        
        assert "failed_tasks" in result
        assert "task_006" in result["failed_tasks"]
        assert "error" in result
        assert "Agent execution failed" in result["error"][0]
        # Budget should still be updated even on failure
        assert result["budget"]["total_searches"] == 5


class TestExtractCitations:
    """Test cases for _extract_citations function."""
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.get_deepseek_chat")
    @patch("app.agents.sub_agent.SUB_AGENT_CITATION_EXTRACTION_TEMPLATE")
    async def test_extract_citations_success_returns_findings(self, mock_template, mock_get_llm):
        """Test successful citation extraction."""
        # Mock LLM chain
        extraction_output = CitationExtractionOutput(
            findings=[
                Finding(
                    claim="Test claim",
                    citation=Citation(source="Test", url="http://test.com"),
                    topic="test",
                    credibility_score=0.8
                )
            ]
        )
        
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = extraction_output
        mock_template.__or__ = MagicMock(return_value=mock_chain)
        
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        result = await _extract_citations(
            raw_results="[Source: tavily_search] Test results",
            topic="test",
            source_tools=["tavily_search"]
        )
        
        assert len(result) == 1
        assert result[0].claim == "Test claim"
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.SUB_AGENT_CITATION_EXTRACTION_TEMPLATE")
    @patch("app.agents.sub_agent.get_deepseek_chat")
    async def test_extract_citations_llm_failure_returns_empty(self, mock_get_llm, mock_template):
        """Test that LLM failures return empty list."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock the chain to raise exception
        mock_chain = AsyncMock()
        mock_chain.ainvoke.side_effect = Exception("LLM error")
        mock_template.__or__.return_value = mock_chain
        
        result = await _extract_citations(
            raw_results="Test results",
            topic="test",
            source_tools=["tool"]
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.get_deepseek_chat")
    @patch("app.agents.sub_agent.SUB_AGENT_CITATION_EXTRACTION_TEMPLATE")
    async def test_extract_citations_empty_results_returns_empty(self, mock_template, mock_get_llm):
        """Test that empty raw results return empty findings."""
        extraction_output = CitationExtractionOutput(findings=[])
        
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = extraction_output
        mock_template.__or__ = MagicMock(return_value=mock_chain)
        
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        result = await _extract_citations(
            raw_results="",
            topic="test",
            source_tools=[]
        )
        
        assert result == []
