import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.api.chat import router, stream_graph_with_tokens
from app.models.schemas import ReviewAction
from fastapi import FastAPI
from langgraph.types import Command

app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestGetOrCreateUser:
    def test_get_user(self):
        response = client.get("/chat/user")
        assert response.status_code == 200
        assert "user_" in response.json()["user_id"]

class TestStreamGraphWithTokens:
    @pytest.mark.asyncio
    @patch("app.api.chat.update_conversation_status", new_callable=AsyncMock)
    @patch("app.api.chat.save_conversation", new_callable=AsyncMock)
    async def test_stream_graph_with_tokens_messages(self, mock_save, mock_update):
        # Mock graph and astream
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=None)
        
        async def mock_astream(*args, **kwargs):
            # yield message from report agent
            msg_mock = MagicMock()
            msg_mock.content = "report content"
            yield ("messages", (msg_mock, {"langgraph_node": "report_agent"}))
            
            # yield message from scope
            msg_mock2 = MagicMock()
            msg_mock2.content = "scope message"
            yield ("messages", (msg_mock2, {"langgraph_node": "scope", "tags": ["user_visible"]}))
            
            # yield updates from supervisor (with task history > 3 and dicts)
            yield ("updates", {
                "supervisor": {
                    "task_history": [{"topic": "Topic 1"}],
                    "budget": {"iterations": 1}
                }
            })
            
            # second round of supervisor to trigger 'Deepening analysis'
            yield ("updates", {
                "supervisor": {
                    "task_history": [MagicMock(topic="Topic 2")],
                    "budget": {"iterations": 2}
                }
            })
            
            # yield updates from supervisor (empty tasks)
            yield ("updates", {
                "supervisor": {
                    "task_history": [],
                    "gaps": {"reasoning": "Need more sources"}
                }
            })
            
            # yield updates from supervisor (object gaps)
            mock_gaps = MagicMock(reasoning="Gap object reason")
            yield ("updates", {
                "supervisor": {
                    "task_history": [],
                    "gaps": mock_gaps
                }
            })

            # yield updates from sub_agent with key insights
            summary = MagicMock()
            summary.key_insights = ["Insight 1"]
            yield ("updates", {
                "sub_agent": {"sub_agent_summaries": [summary]}
            })
            
            # yield updates from sub_agent without key insights (finding count)
            summary2 = MagicMock(key_insights=[], finding_count=2)
            yield ("updates", {
                "sub_agent": {"sub_agent_summaries": [summary2]}
            })

            # yield updates from sub_agent with dict
            yield ("updates", {
                "sub_agent": {"sub_agent_summaries": [{"key_insights": [], "finding_count": 5}]}
            })

            # yield complete
            brief_mock = MagicMock()
            brief_mock.scope = "Mock Scope"
            brief_mock.sub_topics = ["t1"]
            yield ("updates", {
                "report_agent": {
                    "research_brief": dict({"scope": "sc", "sub_topics": []}), # using dict for branch coverage
                    "findings": ["f1"],
                    "report_content": "rep",
                    "is_complete": True
                }
            })
            
            # Yield research brief as Object branch coverage
            yield ("updates", {
                "report_agent": {
                    "research_brief": brief_mock
                }
            })
            
            # test generic non-dict updates
            yield ("updates", {
                "reviewer": {"is_complete": False}
            })
            
            # test dict updates with no mode
            yield {"some_node": {"data": "..."}}
            
            # test scope node
            yield ("updates", {
                "scope": {"gaps": {}}
            })

        mock_graph.astream = mock_astream
        
        config = {"configurable": {"thread_id": "test_thread"}}
        input_data = {}
        
        events = []
        async for event in stream_graph_with_tokens(mock_graph, input_data, config, "q", "u", "th"):
            events.append(event)
            
        events_str = "".join(events)
        assert "report content" in events_str
        assert "scope message" in events_str
        assert "Topic 1" in events_str
        assert "Topic 2" in events_str
        assert "Need more sources" in events_str
        assert "Insight 1" in events_str
        assert "complete" in events_str
        assert "Mock Scope" in events_str
        
        mock_update.assert_called()
        mock_save.assert_called()

    @pytest.mark.asyncio
    async def test_stream_graph_command_resume(self):
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "research_brief": MagicMock(),
            "report_content": "rep",
            "findings": []
        }
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        
        async def mock_astream(*args, **kwargs):
            yield ("updates", {})
        
        mock_graph.astream = mock_astream
        config = {"configurable": {"thread_id": "test_thread"}}
        
        events = []
        async for event in stream_graph_with_tokens(mock_graph, Command(resume="yes"), config):
            events.append(event)
        
        assert len(events) > 0
        assert mock_graph.aget_state.call_count == 2
        
    @pytest.mark.asyncio
    async def test_stream_graph_exception_caught(self):
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=None)
        
        async def mock_astream(*args, **kwargs):
            raise Exception("Stream error")
            yield ("updates", {})
            
        mock_graph.astream = mock_astream
        
        events = []
        async for event in stream_graph_with_tokens(mock_graph, {}, {}):
            events.append(event)
            
        assert any("Stream error" in e for e in events)
        
    @pytest.mark.asyncio
    @patch("app.api.chat.update_conversation_status", new_callable=AsyncMock)
    async def test_stream_graph_interrupt_clarification(self, mock_update):
        mock_graph = MagicMock()
        
        async def mock_astream(*args, **kwargs):
            yield ("updates", {})
            
        mock_graph.astream = mock_astream
        
        mock_state = MagicMock()
        mock_task = MagicMock()
        mock_interrupt = MagicMock()
        mock_interrupt.value = {"type": "clarification_request", "questions": "Q?"}
        mock_task.interrupts = [mock_interrupt]
        mock_state.tasks = [mock_task]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        
        events = []
        async for event in stream_graph_with_tokens(mock_graph, {}, {}):
            events.append(event)
            
        events_str = "".join(events)
        assert "Q?" in events_str
        mock_update.assert_called_with(user_id='default_user', conversation_id='', status='in_progress', phase='scoping')

    @pytest.mark.asyncio
    @patch("app.api.chat.update_conversation_status", new_callable=AsyncMock)
    async def test_stream_graph_interrupt_review(self, mock_update):
        mock_graph = MagicMock()
        
        async def mock_astream(*args, **kwargs):
            yield ("updates", {})
            
        mock_graph.astream = mock_astream
        
        mock_state = MagicMock()
        mock_task = MagicMock()
        mock_interrupt = MagicMock()
        mock_interrupt.value = {"type": "review_request", "report": "REP"}
        mock_task.interrupts = [mock_interrupt]
        mock_state.tasks = [mock_task]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        
        events = []
        async for event in stream_graph_with_tokens(mock_graph, {}, {}):
            events.append(event)
            
        events_str = "".join(events)
        assert "REP" in events_str
        mock_update.assert_called()

class TestChatEndpoint:
    @patch("app.api.chat.build_research_graph")
    @patch("app.api.chat.save_in_progress_conversation", new_callable=AsyncMock)
    def test_post_chat(self, mock_save, mock_build):
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=None)
        mock_build.return_value = mock_graph
        
        req = {
            "message": "Hello", 
            "messages": [{"role": "user", "content": "Prev"}],
            "thread_id": "thread_1",
            "user_id": "u1"
        }
        response = client.post("/chat", json=req)
        assert response.status_code == 200
        mock_build.assert_called_once()
        mock_save.assert_called_once()

    @patch("app.api.chat.build_research_graph")
    def test_post_chat_graph_fail(self, mock_build):
        mock_build.side_effect = Exception("failed initialization")
        req = {"message": "Hello"}
        response = client.post("/chat", json=req)
        assert response.status_code == 500
        
    @patch("app.api.chat.build_research_graph")
    @patch("app.api.chat.save_in_progress_conversation", new_callable=AsyncMock)
    def test_post_chat_existing_interrupt(self, mock_save, mock_build):
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_task = MagicMock()
        mock_interrupt = MagicMock()
        mock_interrupt.value = {"type": "clarification_request"}
        mock_task.interrupts = [mock_interrupt]
        mock_state.tasks = [mock_task]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        mock_build.return_value = mock_graph
        
        req = {"message": "Hello"}
        response = client.post("/chat", json=req)
        assert response.status_code == 200
        mock_save.assert_not_called()

class TestResumeReviewEndpoint:
    @patch("app.api.chat.build_research_graph")
    def test_resume_review_success(self, mock_build):
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {"is_complete": False, "messages": [{"role": "user", "content": "q"}]}
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        mock_build.return_value = mock_graph
        
        response = client.post("/chat/th_1/resume", json={"action": "approve"})
        assert response.status_code == 200
        
    @patch("app.api.chat.build_research_graph")
    @patch("app.api.chat.update_conversation_status", new_callable=AsyncMock)
    def test_resume_review_already_complete(self, mock_update, mock_build):
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.config = {"configurable": {"user_id": "u1"}}
        mock_state.values = {"is_complete": True, "messages": [{"role": "user", "content": "q"}]}
        mock_state.next = []
        mock_state.tasks = []
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        mock_build.return_value = mock_graph
        
        response = client.post("/chat/th_1/resume", json={"action": "approve"})
        assert response.status_code == 200
        mock_update.assert_called_with(user_id='u1', conversation_id='th_1', status='complete', phase='complete')
        
    @patch("app.api.chat.build_research_graph")
    def test_resume_review_graph_fail(self, mock_build):
        mock_build.side_effect = Exception("error")
        response = client.post("/chat/t/resume", json={"action": "approve"})
        assert response.status_code == 500

class TestContinueConversationEndpoint:
    @patch("app.api.chat.build_research_graph")
    def test_continue_success(self, mock_build):
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.config = {"configurable": {"user_id": "ux"}}
        mock_state.values = {"messages": [{"role": "user", "content": "qq"}]}
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        mock_build.return_value = mock_graph
        
        response = client.post("/chat/t/continue")
        assert response.status_code == 200

    @patch("app.api.chat.build_research_graph")
    def test_continue_graph_fail(self, mock_build):
        mock_build.side_effect = Exception("init error")
        response = client.post("/chat/t/continue")
        assert response.status_code == 500
