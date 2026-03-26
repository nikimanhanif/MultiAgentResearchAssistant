"""
Integration tests for the entire backend research workflow.
Matches the 7 key scenarios defined in the implementation plan.
"""

import pytest
import uuid
import os
import aiosqlite
from typing import AsyncGenerator, Any
from unittest.mock import patch, MagicMock, AsyncMock

from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableLambda
from pydantic import Field
import json

import app.persistence.store as app_store
import app.persistence.checkpointer as app_checkpointer

from app.graphs.research_graph import build_research_graph
from app.graphs.state import ResearchState, create_initial_state
from app.models.schemas import (
    ScopeCompletionCheck,
    ResearchBrief,
    ClarificationQuestions,
    ResearchTask,
    ReviewAction,
    Finding,
    Citation,
    SubAgentSummary,
)
from app.agents.supervisor_agent import GapAnalysisOutput
from app.agents.sub_agent import CitationExtractionOutput

# -------------------------------------------------------------------
# Fixtures for Database and Mocks
# -------------------------------------------------------------------

class FakeChatModel(BaseChatModel):
    """Fake chat model for deterministic testing."""
    responses: list[Any] = Field(default_factory=lambda: ["default response"])

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs) -> ChatResult:
        res = self.responses.pop(0) if self.responses else "default response"
        msg = res if isinstance(res, BaseMessage) else AIMessage(content=str(res))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs) -> ChatResult:
        res = self.responses.pop(0) if self.responses else "default response"
        msg = res if isinstance(res, BaseMessage) else AIMessage(content=str(res))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"
    
    def bind_tools(self, *args, **kwargs):
        # LangChain create_agent requires bind_tools to return a runnable
        return self
        
    def with_structured_output(self, schema, **kwargs):
        # Return a runnable that pops from responses and returns it directly
        async def _ainvoke(input_data, config=None, **kwargs):
            res = self.responses.pop(0) if self.responses else None
            # If it's a string, try to parse it (assuming JSON to construct Pydantic)
            if isinstance(res, str) and hasattr(schema, 'model_validate_json'):
                try:
                    return schema.model_validate_json(res)
                except Exception:
                    pass
            return res
            
        def _invoke(input_data, config=None, **kwargs):
            res = self.responses.pop(0) if self.responses else None
            if isinstance(res, str) and hasattr(schema, 'model_validate_json'):
                try:
                    return schema.model_validate_json(res)
                except Exception:
                    pass
            return res
            
        runnable = RunnableLambda(_invoke)
        runnable.ainvoke = _ainvoke
        return runnable

@pytest.fixture
async def mock_db() -> AsyncGenerator[None, None]:
    """
    Setup an in-memory SQLite database for both Store and Checkpointer.
    Patches the global initialized state in the original modules.
    """
    conn_string = "file::memory:?cache=shared"
    conn_store = await aiosqlite.connect(conn_string, uri=True, isolation_level=None)
    conn_cp = await aiosqlite.connect(conn_string, uri=True, isolation_level=None)
    
    test_store = AsyncSqliteStore(conn_store)
    await test_store.setup()
    
    test_checkpointer = AsyncSqliteSaver(conn_cp)
    await test_checkpointer.setup()
    
    # Patch the global references in app.persistence to use our memory ones
    with patch("app.persistence.store._store", test_store), \
         patch("app.persistence.checkpointer._checkpointer", test_checkpointer):
        
        yield
        
    await conn_store.close()
    await conn_cp.close()

@pytest.fixture
def mock_llms():
    """
    Patches the LLM factory functions in all agent modules.
    Test cases can configure the return values of these mocks individually.
    """
    chat_instance = FakeChatModel()
    reasoner_instance = FakeChatModel()
    reasoner_json_instance = FakeChatModel()

    with patch("app.agents.scope_agent.get_deepseek_chat", return_value=chat_instance), \
         patch("app.agents.sub_agent.get_deepseek_chat", return_value=chat_instance), \
         patch("app.agents.supervisor_agent.get_deepseek_reasoner_json", return_value=reasoner_json_instance), \
         patch("app.agents.report_agent.get_deepseek_reasoner", return_value=reasoner_instance):
        
        yield {
            "chat": chat_instance,
            "reasoner": reasoner_instance,
            "reasoner_json": reasoner_json_instance
        }

@pytest.fixture
def mock_tools():
    """
    Patches the academic search functions internal methods.
    """
    with patch("app.tools.academic.arxiv._search_arxiv_internal") as mock_arxiv, \
         patch("app.tools.academic.semantic_scholar._search_semantic_scholar_sync") as mock_ss, \
         patch("app.tools.academic.scopus._search_scopus_sync") as mock_scopus:
        
        # Set default empty returns
        mock_arxiv.return_value = []
        mock_ss.return_value = []
        mock_scopus.return_value = []
        
        yield {
            "arxiv": mock_arxiv,
            "semantic_scholar": mock_ss,
            "scopus": mock_scopus
        }

@pytest.fixture
def compiled_graph(mock_db):
    """Returns a compiled instance of the research graph using mocked DB."""
    return build_research_graph()

# -------------------------------------------------------------------
# Test Scenario Stubs
# -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vague_query_clarification_flow(compiled_graph, mock_llms):
    """Test 1: Vague query -> clarification -> brief creation."""
    # Set up configuration and initial state
    config = {"configurable": {"thread_id": "test_workflow_1", "user_id": "test_user"}}
    initial_state = {
        "messages": [{"role": "user", "content": "Vague query about AI"}],
        "budget": {
            "iterations": 0,
            "max_iterations": 20,
            "max_sub_agents": 20,
            "max_searches_per_agent": 2,
            "total_searches": 0
        }
    }
    
    # 1. First run: The graph should evaluate the completion check.
    # We mock the completion check to return incomplete, and question gen to return questions.
    # Setting ChatDeepSeek mock response (for both check and questions):
    # Depending on what runs first, we might need a dynamic answer, but simpler:
    # check_scope_completion uses ChatDeepSeek. generate_clarification_questions uses ChatDeepSeek.
    # Actually they both use the same model instance, but different prompts.
    # To keep things simple, we'll patch the agent functions directly for this test loop
    # or just supply a JSON string that works for both. 
    # Wait, check_scope_completion needs JSON matching ScopeCompletionCheck.
    # generate_clarification_questions just needs a string.
    # If the LLM returns JSON, generate_clarification_questions will return that JSON string.
    
    mock_llms["chat"].responses = [
        '{"is_complete": false, "reasoning": "Need more info", "missing_info": ["details"], "clarification_questions": [{"question": "What specifically?"}], "context": "Need details"}',
        "What specifically?"
    ]
    
    # Run the graph
    events = []
    async for event in compiled_graph.astream(initial_state, config):
        events.append(event)
        
    # Get current state
    state = await compiled_graph.aget_state(config)
    
    # The graph should be interrupted at scope_wait
    assert len(state.next) > 0
    assert state.next[0] == "scope_wait"
    
    # Verify state contains pending questions
    assert state.values.get("pending_clarification_questions") is not None
    assert "What specifically?" in state.values["pending_clarification_questions"]
    
    # 2. Second run: Resume the graph with user's clarification answer.
    # Now we want it to be considered complete, so it generates a brief.
    mock_llms["chat"].responses = [
        '{"is_complete": true, "reasoning": "Got details", "missing_info": [], "scope": "AI in Healthcare", "sub_topics": ["ML", "Data"], "constraints": {}, "deliverables": "Report"}',
        '{"scope": "AI in Healthcare", "sub_topics": ["ML", "Data"], "constraints": {}, "deliverables": "Report", "format": "deep_research"}'
    ]
    
    # Provide dummy responses for the rest of the flow to avoid errors
    mock_llms["reasoner_json"].responses = ['{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Done"}']
    mock_llms["reasoner"].responses = ["Final report"]

    from langgraph.types import Command
    
    async for event in compiled_graph.astream(Command(resume="I want to focus on Healthcare"), config):
        pass
        
    # Get final state
    state = await compiled_graph.aget_state(config)
    
    # The brief should be populated
    brief = state.values.get("research_brief")
    assert brief is not None
    assert brief.scope == "AI in Healthcare"
    
    # Next should transition to reviewer since we mocked the rest of the flow to complete immediately
    assert len(state.next) > 0
    assert state.next[0] == "reviewer"

@pytest.mark.asyncio
async def test_end_to_end_research_flow(compiled_graph, mock_llms, mock_tools):
    """Test 2: Brief -> supervisor tasking -> sub-agent findings -> report."""
    config = {"configurable": {"thread_id": "test_workflow_2", "user_id": "test_user"}}
    
    # Init state directly with a brief to skip scope agent
    brief = ResearchBrief(
        scope="Quantum Computing",
        sub_topics=["Qubits"],
        constraints={},
        deliverables="Markdown report",
        format="deep_research"
    )
    initial_state = {
        "research_brief": brief,
        "messages": [],
        "budget": {
            "iterations": 0,
            "max_iterations": 3,
            "max_sub_agents": 2,
            "max_searches_per_agent": 1,
            "total_searches": 0
        }
    }
    
    # Mocks for supervisor (iteration 1 -> generate tasks)
    tasks_json = '[{"task_id": "t1", "description": "Research Qubits", "query_variants": ["qubits"]}]'
    # Second iteration supervisor call
    gaps_json = '{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Tasks done"}'
    
    mock_llms["reasoner"].responses = ["# Final Report on Quantum Computing"]
    mock_llms["reasoner_json"].responses = [tasks_json, gaps_json]
    
    # Mock tool response
    mock_tools["arxiv"].return_value = [{
        "source": "arxiv",
        "paper_id": "1234.5678",
        "title": "Qubits paper",
        "authors": "John Doe",
        "abstract": "Info about qubits",
        "year": 2024,
        "pdf_url": "http://arxiv.org/pdf/1234.5678",
        "citation_count": None
    }]
    
    # Mocks for sub_agent (iteration 1 -> single task t1)
    # The LangChain agent needs a tool call, then a final response.
    tool_call_msg = AIMessage(
        content="", 
        tool_calls=[{"name": "search_arxiv", "args": {"query": "qubits"}, "id": "call_abc"}]
    )
    # Finally, _extract_citations will parse structured output
    citation_output = CitationExtractionOutput(
        findings=[
            Finding(
                claim="Qubits are quantum bits",
                topic="Qubits",
                credibility_score=0.9,
                citation=Citation(source="ArXiv", title="Qubits paper", authors=["John Doe"])
            )
        ],
        task_answered=True,
        key_insights=["Qubits basic"],
        gaps_noted=None
    )
    
    mock_llms["chat"].responses = [
        tool_call_msg,
        "I found some info on ArXiv.",  # agent final response
        citation_output  # _extract_citations response
    ]
    

    
    # Run graph until it pauses (which should be at reviewer)
    async for event in compiled_graph.astream(initial_state, config):
        pass
        
    state = await compiled_graph.aget_state(config)
    
    # Check that it paused at reviewer
    assert len(state.next) > 0
    assert state.next[0] == "reviewer"
    
    # Verify task was generated and completed
    assert len(state.values.get("completed_tasks", [])) == 1
    assert state.values["completed_tasks"][0] == "t1"
    
    # Verify findings were populated
    findings = state.values.get("findings", [])
    assert len(findings) == 1
    assert findings[0].claim == "Qubits are quantum bits"
    
    # Verify report was generated
    assert state.values.get("report_content") == "# Final Report on Quantum Computing"
    
    # Verify tool was called
    mock_tools["arxiv"].assert_called_once()

@pytest.mark.asyncio
@pytest.mark.parametrize("review_action, expected_next_node", [
    ("approve", "__end__"),
    ("refine", "report_agent"),
    ("re_research", "supervisor")
])
async def test_reviewer_branches(compiled_graph, mock_llms, review_action, expected_next_node):
    """Test 3: Reviewer approve / refine / re-research branches."""
    config = {"configurable": {"thread_id": f"test_reviewer_{review_action}", "user_id": "test_user"}}
    
    # Init state with is_complete=True so supervisor routes directly to report_agent
    brief = ResearchBrief(
        scope="Quantum Computing",
        sub_topics=["Qubits"],
        constraints={},
        deliverables="Markdown report",
        format="deep_research"
    )
    
    initial_state = {
        "research_brief": brief,
        "is_complete": True,
        "budget": {
            "iterations": 1, 
            "max_iterations": 3,
            "max_sub_agents": 2,
            "max_searches_per_agent": 1,
            "total_searches": 0
        },
        "findings": [
            Finding(
                claim="Qubits are quantum bits",
                topic="Qubits",
                credibility_score=0.9,
                citation=Citation(source="ArXiv", title="Qubits paper", authors=["John Doe"])
            )
        ]
    }
    
    # Both supervisor (gap analysis) and report agent will run.
    mock_llms["reasoner_json"].responses = [
        '{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Done"}'
    ]
    mock_llms["reasoner"].responses = ["# Draft Report"]
    
    # Run graph until it pauses at reviewer
    async for event in compiled_graph.astream(initial_state, config):
        pass
        
    state = await compiled_graph.aget_state(config)
    assert len(state.next) > 0 and state.next[0] == "reviewer"
    
    # Prepare responses for the resume if it loops back
    mock_llms["reasoner"].responses = ["# Refined Report"]
    
    if review_action == "re_research":
        # Supervisor runs on re-research. Return is_complete=True to bypass sub_agents.
        mock_llms["reasoner_json"].responses = [
            '{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Done"}'
        ]
        
    from langgraph.types import Command
    
    # Resume with the reviewer action
    async for event in compiled_graph.astream(Command(resume={"action": review_action, "feedback": "Needs work"}), config):
        pass
        
    state = await compiled_graph.aget_state(config)
    
    if expected_next_node == "__end__":
        assert len(state.next) == 0
        assert state.values.get("is_complete") is True
    else:
        # Both refine and re_research loop back to the reviewer interrupt
        assert len(state.next) > 0
        assert state.next[0] == "reviewer"
        
        # Verify Report was regenerated
        assert state.values.get("report_content") == "# Refined Report"
        
        if review_action == "refine":
            # The feedback is cleared by report_agent, so we just verify the state looped back
            pass
        elif review_action == "re_research":
            # Verify new task was added by reviewer
            assert len(state.values.get("task_history", [])) > 0
            assert state.values["task_history"][-1].requested_by == "reviewer"

@pytest.mark.asyncio
async def test_persistence_and_resume(mock_db, mock_llms):
    """Test 4: Persistence and resume across interrupts with a new graph instance."""
    config = {"configurable": {"thread_id": "test_persistence", "user_id": "test_user"}}
    
    # 1. First graph instance
    graph1 = build_research_graph()
    initial_state = {
        "messages": [{"role": "user", "content": "Vague task"}],
        "budget": {
            "iterations": 0,
            "max_iterations": 3,
            "max_sub_agents": 2,
            "max_searches_per_agent": 1,
            "total_searches": 0
        }
    }
    
    # Mock scope agent to ask a clarification question
    mock_llms["chat"].responses = [
        '{"is_complete": false, "reasoning": "Need more info", "missing_info": ["details"], "clarification_questions": [{"question": "Details?"}], "context": "None"}',
        "Details?" # For generate_clarification_questions
    ]
    
    # Run until interrupt
    async for event in graph1.astream(initial_state, config):
        pass
        
    state1 = await graph1.aget_state(config)
    assert state1.next[0] == "scope_wait"
    
    # 2. Second graph instance (simulating server restart/stateless API)
    graph2 = build_research_graph()
    
    # The state should be exactly the same
    state2 = await graph2.aget_state(config)
    assert len(state2.tasks) > 0
    assert state2.next[0] == "scope_wait"
    assert "Details?" in state2.values.get("pending_clarification_questions", [])
    
    # 3. Resume using the second graph instance
    mock_llms["chat"].responses = [
        '{"is_complete": true, "reasoning": "Got details", "missing_info": [], "scope": "AI", "sub_topics": ["ML"], "constraints": {}, "deliverables": "Report"}',
        '{"scope": "AI", "sub_topics": ["ML"], "constraints": {}, "deliverables": "Report", "format": "deep_research"}'
    ]
    # Provide dummy for supervisor to avoid errors
    mock_llms["reasoner_json"].responses = ['{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Done"}']
    mock_llms["reasoner"].responses = ["# Report"]
    
    from langgraph.types import Command
    async for event in graph2.astream(Command(resume="Here are details"), config):
        pass
        
    state3 = await graph2.aget_state(config)
    # Graph should reach the reviewer interrupt
    assert state3.next[0] == "reviewer"
    assert state3.values.get("research_brief") is not None


@pytest.mark.asyncio
async def test_conversation_retrieval(mock_db, mock_llms):
    """Test 5: Conversation retrieval and state inspection mid-workflow."""
    config = {"configurable": {"thread_id": "test_retrieval", "user_id": "test_user"}}
    graph = build_research_graph()
    
    initial_state = {
        "messages": [AIMessage(content="First message from AI")],
        "budget": {"iterations": 0, "max_iterations": 3}
    }
    
    # Just update the state directly to simulate a run
    await graph.aupdate_state(config, initial_state)
    
    # Retrieve the state
    state = await graph.aget_state(config)
    messages = state.values.get("messages", [])
    
    assert len(messages) == 1
    assert messages[0].content == "First message from AI"
    # End test 5

@pytest.mark.asyncio
async def test_invalid_citation_warning_appended(compiled_graph, mock_llms, mock_tools):
    """Test 6: Invalid citation index warning appended to report."""
    config = {"configurable": {"thread_id": "test_invalid_citation", "user_id": "test_user"}}
    
    brief = ResearchBrief(
        scope="Test Scope", sub_topics=["T1"], constraints={},
        deliverables="Report", format="deep_research"
    )
    
    initial_state = {
        "research_brief": brief,
        "is_complete": True,
        "budget": {"iterations": 1, "max_iterations": 3, "max_sub_agents": 2, "max_searches_per_agent": 1, "total_searches": 0},
        "findings": [
            Finding(
                claim="Valid", topic="T1", credibility_score=0.9,
                citation=Citation(source="ArXiv", title="Valid Paper", authors=["Author"])
            )
        ]
    }
    
    # Return a report that cites [1] (valid) and [2] (invalid, doesn't exist)
    mock_llms["reasoner"].responses = ["# Report\n\nHere is a valid claim [1]. Here is a hallucinated claim [2]."]
    
    async for event in compiled_graph.astream(initial_state, config):
        pass
        
    state = await compiled_graph.aget_state(config)
    report = state.values.get("report_content", "")
    
    # The report agent should have detected the invalid [2] citation and appended a warning
    assert "[2]" in report
    assert "Citation Warning" in report
    assert "hallucinated" in report.lower()

@pytest.mark.asyncio
async def test_duplicate_findings_merged(compiled_graph, mock_llms, mock_tools):
    """Test 7: Duplicate findings merged correctly across parallel tasks."""
    config = {"configurable": {"thread_id": "test_duplicates", "user_id": "test_user"}}
    
    brief = ResearchBrief(
        scope="Test Scope", sub_topics=["T1", "T2"], constraints={},
        deliverables="Report", format="deep_research"
    )
    
    # Start at supervisor with 2 pending tasks so 2 sub-agents spawn in parallel
    initial_state = {
        "research_brief": brief,
        "is_complete": False,
        "findings": [],
        "budget": {"iterations": 1, "max_iterations": 3, "max_sub_agents": 4, "max_searches_per_agent": 1, "total_searches": 0},
        "task_history": [
            ResearchTask(task_id="t1", topic="T1", query="q1", priority=1, requested_by="sup"),
            ResearchTask(task_id="t2", topic="T2", query="q2", priority=1, requested_by="sup")
        ]
    }
    
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "search_arxiv", "args": {"query": "q"}, "id": "call_123"}]
    )
    
    output = CitationExtractionOutput(
        findings=[
            Finding(
                claim="Same claim", topic="T1", credibility_score=0.9,
                citation=Citation(source="ArXiv", title="Same Paper", authors=["John Doe"], doi="10.1234/same")
            )
        ],
        task_answered=True, key_insights=["Insight"], gaps_noted=None
    )
    
    # Since parallel execution with a shared mock responses list causes race conditions,
    # we patch the node itself.
    mock_findings = [
        Finding(
            claim="Same claim", topic="T1", credibility_score=0.9,
            citation=Citation(source="ArXiv", title="Same Paper", authors=["John Doe"], doi="10.1234/same")
        )
    ]
    
    async def mock_sub_agent_node(state):
        return {
            "findings": mock_findings,
            "completed_tasks": [state["task"].task_id],
            "sub_agent_summaries": [SubAgentSummary(task_id=state["task"].task_id, task_answered=True, finding_count=1, key_insights=["..."])]
        }

    with patch("app.graphs.research_graph.sub_agent_node", side_effect=mock_sub_agent_node):
        # Re-build graph inside patch so it uses the mock
        graph = build_research_graph()
        
        # Supervisor will run twice: 
        # 1. First to route to sub-agents.
        # 2. Second to aggregate after sub-agents return.
        mock_llms["reasoner_json"].responses = [
            '{"has_gaps": false, "is_complete": false, "gaps_identified": [], "new_tasks": [], "reasoning": "Working"}',
            '{"has_gaps": false, "is_complete": true, "gaps_identified": [], "new_tasks": [], "reasoning": "Done"}'
        ]
        # report_agent needs a response
        mock_llms["reasoner"].responses = ["# Report"]
        
        # Run graph
        async for event in graph.astream(initial_state, config):
            pass
        
        state = await graph.aget_state(config)
    
    # Check findings length. Because they share the same DOI, they should be deduplicated to 1.
    findings = state.values.get("findings", [])
    assert len(findings) == 1
    assert findings[0].claim == "Same claim"
