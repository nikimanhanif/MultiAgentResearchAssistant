"""Unit tests for Sub-Agent node."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.models.schemas import ResearchTask, Finding, Citation, SourceType
from app.agents.sub_agent import (
    sub_agent_node,
    _extract_citations,
    CitationExtractionOutput
)
from langchain_core.messages import ToolMessage, AIMessage
from app.graphs.state import SubAgentState
from app.models.schemas import ResearchBrief
from langgraph.errors import GraphRecursionError


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
    @patch("app.agents.sub_agent.create_agent")
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
        
        mock_agent = AsyncMock()
        
        mock_ai_msg = MagicMock(spec=AIMessage)
        mock_ai_msg.tool_calls = [{"name": "tavily_search", "args": {}, "id": "call_1"}]
        
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
        
        finding = Finding(
            claim="Quantum computing is advancing",
            citation=Citation(source="Nature", url="http://nature.com/1"),
            topic="quantum computing",
            credibility_score=0.9
        )
        mock_extract.return_value = CitationExtractionOutput(
            findings=[finding],
            task_answered=True,
            key_insights=["Quantum computing is advancing [0]"],
            gaps_noted=None
        )
        
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
        assert "sub_agent_summaries" in result
        assert len(result["sub_agent_summaries"]) == 1
        assert result["sub_agent_summaries"][0].task_id == "task_004"
        assert "budget" in result
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.get_research_tools")
    @patch("app.agents.sub_agent.get_deepseek_chat")
    @patch("app.agents.sub_agent.create_agent")
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
        assert result["budget"]["total_searches"] == 5

    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.ls.get_current_run_tree")
    @patch("app.agents.sub_agent.get_research_tools")
    @patch("app.agents.sub_agent.get_deepseek_chat")
    @patch("app.agents.sub_agent.create_agent")
    async def test_sub_agent_node_run_tree_metadata_and_recursion_error(
        self, mock_create_agent, mock_get_llm, mock_get_tools, mock_run_tree
    ):
        # Test lines 86-88 (metadata) and 161-165 (GraphRecursionError)
        mock_rt = MagicMock()
        mock_rt.metadata = {}
        mock_run_tree.return_value = mock_rt
        
        mock_tool = MagicMock()
        mock_tool.name = "tavily_search"
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = [mock_tool]
        mock_ctx.__aexit__.return_value = None
        mock_get_tools.return_value = mock_ctx
        
        mock_agent = AsyncMock()
        error = GraphRecursionError()
        error.state = {"messages": []}
        mock_agent.ainvoke.side_effect = error
        mock_create_agent.return_value = mock_agent

        state: SubAgentState = {
            "task": ResearchTask(
                task_id="task_rec",
                topic="test topic rec",
                query="test query rec",
                priority=1
            ),
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["test topic rec"],
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
        
        assert mock_rt.metadata["task_id"] == "task_rec"
        assert mock_rt.metadata["topic"] == "test topic rec"
        assert mock_rt.metadata["priority"] == 1
        assert "failed_tasks" in result
        assert "task_rec" in result["failed_tasks"]
        assert "error" in result
        assert "hit recursion limit with no results" in result["error"][0]
        
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent._extract_citations")
    @patch("app.agents.sub_agent.create_agent")
    @patch("app.agents.sub_agent.get_research_tools")
    @patch("app.agents.sub_agent.get_deepseek_chat")
    async def test_sub_agent_node_unknown_tool_and_no_tool_results(
        self, mock_get_llm, mock_get_tools, mock_create_agent, mock_extract
    ):
        # Test lines 178-184 (unknown tool name lookup) and then test 194 (no tool results)
        mock_tool = MagicMock()
        mock_tool.name = "tavily_search"
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = [mock_tool]
        mock_ctx.__aexit__.return_value = None
        mock_get_tools.return_value = mock_ctx
        
        mock_agent = AsyncMock()
        mock_ai_msg = MagicMock(spec=AIMessage)
        mock_ai_msg.tool_calls = [{"name": "real_tool", "args": {}, "id": "call_2"}]
        mock_tool_msg = MagicMock(spec=ToolMessage)
        mock_tool_msg.name = "" # trigger unknown
        mock_tool_msg.content = "Search results"
        mock_tool_msg.tool_call_id = "call_2"
        
        mock_agent.ainvoke.return_value = {"messages": [mock_ai_msg, mock_tool_msg]}
        mock_create_agent.return_value = mock_agent
        
        mock_extract.return_value = CitationExtractionOutput(
            findings=[],
            task_answered=True,
            key_insights=[],
            gaps_noted=None
        )

        state: SubAgentState = {
            "task": ResearchTask(
                task_id="task_unk",
                topic="topic",
                query="query",
                priority=1
            ),
            "research_brief": ResearchBrief(
                scope="Test",
                sub_topics=["topic"],
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
        # If the tool name was successfully resolved to "real_tool", it should proceed to _extract_citations
        # Because we set extraction_result.findings=[], it should return state update with findings=[]
        mock_extract.assert_called_once()
        kwargs = mock_extract.call_args.kwargs
        assert kwargs["source_tools"] == ["real_tool"] # Asserts the line 178-184 fix worked

        # Now test line 194 (no tool results)
        mock_agent.ainvoke.return_value = {"messages": [mock_ai_msg]} # No tool msg
        result2 = await sub_agent_node(state)
        assert "failed_tasks" in result2
        assert "task_unk" in result2["failed_tasks"]

class TestExtractCitations:
    """Test cases for _extract_citations function."""
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.get_deepseek_chat")
    @patch("app.agents.sub_agent.SUB_AGENT_CITATION_EXTRACTION_TEMPLATE")
    async def test_extract_citations_success_returns_findings(self, mock_template, mock_get_llm):
        """Test successful citation extraction."""
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
            task_query="test query",
            source_tools=["tavily_search"]
        )
        
        assert len(result.findings) == 1
        assert result.findings[0].claim == "Test claim"
    
    @pytest.mark.asyncio
    @patch("app.agents.sub_agent.SUB_AGENT_CITATION_EXTRACTION_TEMPLATE")
    @patch("app.agents.sub_agent.get_deepseek_chat")
    async def test_extract_citations_llm_failure_returns_empty(self, mock_get_llm, mock_template):
        """Test that LLM failures return empty list."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_chain = AsyncMock()
        mock_chain.ainvoke.side_effect = Exception("LLM error")
        mock_template.__or__.return_value = mock_chain
        
        result = await _extract_citations(
            raw_results="Test results",
            topic="test",
            task_query="test query",
            source_tools=["tool"]
        )
        
        assert result.findings == []
        assert result.task_answered == False
    
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
            task_query="test query",
            source_tools=[]
        )
        
        assert result.findings == []
