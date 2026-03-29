import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.api.conversations import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestListUserConversations:
    @patch('app.api.conversations.list_conversations', new_callable=AsyncMock)
    def test_list_user_conversations_success(self, mock_list):
        mock_list.return_value = [
            {"conversation_id": "1", "user_query": "q1", "created_at": "t1", "status": "complete", "phase": "p1"},
            {"conversation_id": "2", "user_query": "q2", "created_at": "t2"}
        ]
        
        response = client.get("/conversations/user123?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["conversation_id"] == "1"
        assert data[1]["status"] == "complete"

class TestGetConversationDetail:
    @patch('app.api.conversations.get_conversation', new_callable=AsyncMock)
    @patch('app.api.conversations.build_research_graph')
    def test_get_conversation_detail_success(self, mock_build, mock_get):
        mock_get.return_value = {
            "conversation_id": "1",
            "user_query": "q1",
            "created_at": "t1",
            "report_content": "rep",
            "findings": ["f1"]
        }
        
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "messages": [
                {"role": "user", "content": "u1"},
                MagicMock(type="ai", content="a1"),
                MagicMock(type="human", content="h1"),
                MagicMock(type="unknown", content="u1"),
                {"role": "system", "content": "s2"}
            ]
        }
        mock_state.tasks = []
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        mock_build.return_value = mock_graph
        
        response = client.get("/conversations/user123/1")
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "1"
        assert data["findings_count"] == 1
        assert len(data["messages"]) == 3
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][2]["role"] == "user"

    @patch('app.api.conversations.get_conversation', new_callable=AsyncMock)
    def test_get_conversation_detail_not_found(self, mock_get):
        mock_get.return_value = None
        response = client.get("/conversations/user123/1")
        assert response.status_code == 404

    @patch('app.api.conversations.get_conversation', new_callable=AsyncMock)
    @patch('app.api.conversations.build_research_graph')
    def test_get_conversation_detail_with_interrupts(self, mock_build, mock_get):
        mock_get.return_value = {
            "conversation_id": "1", "user_query": "q", "created_at": "t"
        }
        
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_state.values = {
            "messages": [
                {"role": "assistant", "content": "Q0?"}
            ]
        }
        
        # Test distinct interrupt added
        mock_task = MagicMock()
        mock_interrupt = MagicMock()
        mock_interrupt.value = {"type": "clarification_request", "questions": "Q1?"}
        mock_task.interrupts = [mock_interrupt]
        
        # Test duplicate interrupt skipped
        mock_task2 = MagicMock()
        mock_interrupt2 = MagicMock()
        mock_interrupt2.value = {"type": "clarification_request", "questions": "Q1?"}
        mock_task2.interrupts = [mock_interrupt2]
        
        mock_state.tasks = [mock_task, mock_task2]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        mock_build.return_value = mock_graph
        
        response = client.get("/conversations/user123/1")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "Q0?"
        assert data["messages"][1]["content"] == "Q1?"

    @patch('app.api.conversations.get_conversation', new_callable=AsyncMock)
    @patch('app.api.conversations.build_research_graph')
    def test_get_conversation_detail_graph_error_fallback(self, mock_build, mock_get):
        mock_get.return_value = {
            "conversation_id": "1", "user_query": "q", "created_at": "t"
        }
        
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(side_effect=Exception("state error"))
        mock_build.return_value = mock_graph
        
        # Should not fail request
        response = client.get("/conversations/user123/1")
        assert response.status_code == 200
        assert len(response.json()["messages"]) == 0

class TestGetConversationState:
    @patch('app.api.conversations.get_conversation', new_callable=AsyncMock)
    @patch('app.api.conversations.build_research_graph')
    def test_get_conversation_state_in_progress(self, mock_build, mock_get):
        mock_get.return_value = {"status": "in_progress", "phase": "p"}
        
        mock_graph = MagicMock()
        mock_state = MagicMock()
        mock_task = MagicMock()
        mock_interrupt = MagicMock()
        mock_interrupt.value = {"type": "something", "report": "rep content"}
        mock_task.interrupts = [mock_interrupt]
        mock_state.tasks = [mock_task]
        mock_graph.aget_state = AsyncMock(return_value=mock_state)
        mock_build.return_value = mock_graph
        
        response = client.get("/conversations/user123/1/state")
        assert response.status_code == 200
        data = response.json()
        assert data["has_pending_interrupt"] is True
        assert data["interrupt_type"] == "something"
        assert data["report_content"] == "rep content"

    @patch('app.api.conversations.get_conversation', new_callable=AsyncMock)
    def test_get_conversation_state_not_found(self, mock_get):
        mock_get.return_value = None
        response = client.get("/conversations/u/1/state")
        assert response.status_code == 404
        
    @patch('app.api.conversations.get_conversation', new_callable=AsyncMock)
    @patch('app.api.conversations.build_research_graph')
    def test_get_conversation_state_graph_error(self, mock_build, mock_get):
        mock_get.return_value = {"status": "in_progress", "phase": "p"}
        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(side_effect=Exception("API error"))
        mock_build.return_value = mock_graph
        
        # Exception caught cleanly
        response = client.get("/conversations/u/1/state")
        assert response.status_code == 200

class TestDeleteConversation:
    @patch('app.api.conversations.get_store')
    @patch('app.api.conversations.get_checkpointer')
    def test_delete_success(self, mock_check, mock_store):
        store_inst = MagicMock()
        store_inst.aget = AsyncMock(return_value={"id": "1"})
        store_inst.adelete = AsyncMock()
        mock_store.return_value = store_inst
        
        check_inst = MagicMock()
        check_inst.conn.execute = AsyncMock()
        mock_check.return_value = check_inst
        
        response = client.delete("/conversations/user123/1")
        assert response.status_code == 200
        store_inst.adelete.assert_called_once()
        assert check_inst.conn.execute.call_count == 2
        
    @patch('app.api.conversations.get_store')
    def test_delete_not_found(self, mock_store):
        store_inst = MagicMock()
        store_inst.aget = AsyncMock(return_value=None)
        mock_store.return_value = store_inst
        
        response = client.delete("/conversations/u/1")
        assert response.status_code == 404

    @patch('app.api.conversations.get_store')
    @patch('app.api.conversations.get_checkpointer')
    def test_delete_checkpoint_error(self, mock_check, mock_store):
        store_inst = MagicMock()
        store_inst.aget = AsyncMock(return_value={"id": "1"})
        store_inst.adelete = AsyncMock()
        mock_store.return_value = store_inst
        
        check_inst = MagicMock()
        check_inst.conn.execute = AsyncMock(side_effect=Exception("db error"))
        mock_check.return_value = check_inst
        
        # Should gracefully fail checkpointer deletion
        response = client.delete("/conversations/user123/1")
        assert response.status_code == 200

class TestUpdateThinkingState:
    @patch('app.api.conversations.update_thinking_state', new_callable=AsyncMock)
    def test_update_thinking_success(self, mock_update):
        mock_update.return_value = True
        response = client.patch(
            "/conversations/u/1/thinking",
            json={"thinking_state": {"agent_type": "sub_agent"}}
        )
        assert response.status_code == 200
        
    @patch('app.api.conversations.update_thinking_state', new_callable=AsyncMock)
    def test_update_thinking_not_found(self, mock_update):
        mock_update.return_value = False
        response = client.patch(
            "/conversations/u/1/thinking",
            json={"thinking_state": {}}
        )
        assert response.status_code == 404
