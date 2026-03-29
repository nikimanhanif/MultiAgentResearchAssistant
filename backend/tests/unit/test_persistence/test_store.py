"""Unit tests for store persistence module."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime
from app.persistence.store import (
    initialize_store,
    shutdown_store,
    get_store,
    save_conversation,
    get_conversation,
    list_conversations,
    save_in_progress_conversation,
    update_conversation_status,
    update_thinking_state,
)
from app.models.schemas import ResearchBrief, Finding, Citation, ReportFormat

# Reset global state before each test
@pytest.fixture(autouse=True)
async def reset_store():
    await shutdown_store()
    yield
    await shutdown_store()

class TestStore:
    """Test suite for store management and operations."""

    @pytest.mark.asyncio
    async def test_initialize_store_success(self):
        """Test successful initialization of store."""
        with patch("aiosqlite.connect", new_callable=AsyncMock) as mock_connect:
            with patch("app.persistence.store.AsyncSqliteStore") as mock_store_cls:
                mock_conn = MagicMock()
                mock_conn.close = AsyncMock()
                mock_connect.return_value = mock_conn
                
                mock_store = MagicMock()
                mock_store.setup = AsyncMock()
                mock_store.conn = mock_conn
                mock_store_cls.return_value = mock_store
                
                result = await initialize_store()
                
                assert result == mock_store
                mock_connect.assert_called_once()
                mock_store_cls.assert_called_once_with(mock_conn)
                mock_store.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_store_closes_connection(self):
        """Test that shutdown closes the database connection."""
        mock_conn = MagicMock()
        mock_conn.close = AsyncMock()
        mock_store = MagicMock()
        mock_store.conn = mock_conn
        mock_store.setup = AsyncMock()
        
        with patch("aiosqlite.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_conn
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                await shutdown_store()
                
                mock_conn.close.assert_awaited_once()
                
                # Verify it's cleared
                with pytest.raises(RuntimeError, match="Store not initialized"):
                    get_store()

    def test_get_store_raises_if_not_initialized(self):
        """Test that get_store raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="Store not initialized"):
            get_store()

    @pytest.mark.asyncio
    async def test_save_conversation_calls_store_put(self):
        """Test that save_conversation calls store.aput with correct data."""
        mock_store = MagicMock()
        mock_store.aget = AsyncMock(return_value=None)
        mock_store.aput = AsyncMock()
        mock_store.setup = AsyncMock()
        mock_store.conn.close = AsyncMock()
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                brief = ResearchBrief(
                    scope="Test",
                    sub_topics=["t1"],
                    constraints={},
                    deliverables="d1",
                    format=ReportFormat.LITERATURE_REVIEW
                )
                findings = [
                    Finding(
                        claim="c1",
                        citation=Citation(source="s1", url="u1"),
                        topic="t1",
                        credibility_score=0.9
                    )
                ]
                
                await save_conversation(
                    user_id="user1",
                    conversation_id="conv1",
                    user_query="query",
                    research_brief=brief,
                    findings=findings,
                    report_content="report"
                )
                
                mock_store.aput.assert_awaited_once()
                call_args = mock_store.aput.call_args
                assert call_args.kwargs["namespace"] == ("user1", "conversations")
                assert call_args.kwargs["key"] == "conv1"
                assert call_args.kwargs["value"]["user_query"] == "query"
                assert call_args.kwargs["value"]["report_content"] == "report"

    @pytest.mark.asyncio
    async def test_get_conversation_calls_store_get(self):
        """Test that get_conversation calls store.aget."""
        mock_store = MagicMock()
        mock_store.aget = AsyncMock()
        mock_store.setup = AsyncMock()
        mock_store.conn.close = AsyncMock()
        mock_item = MagicMock()
        mock_item.value = {"data": "test"}
        mock_store.aget.return_value = mock_item
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                result = await get_conversation("user1", "conv1")
                
                assert result == {"data": "test"}
                mock_store.aget.assert_awaited_once_with(("user1", "conversations"), "conv1")

    @pytest.mark.asyncio
    async def test_get_conversation_returns_none_if_not_found(self):
        """Test that get_conversation returns None if store returns None."""
        mock_store = MagicMock()
        mock_store.aget = AsyncMock(return_value=None)
        mock_store.setup = AsyncMock()
        mock_store.conn.close = AsyncMock()
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                result = await get_conversation("user1", "conv1")
                
                assert result is None

    @pytest.mark.asyncio
    async def test_list_conversations_calls_store_search(self):
        """Test that list_conversations calls store.asearch and formats results."""
        mock_store = MagicMock()
        mock_store.asearch = AsyncMock()
        mock_store.setup = AsyncMock()
        mock_store.conn.close = AsyncMock()
        
        mock_item1 = MagicMock()
        mock_item1.key = "conv1"
        mock_item1.value = {"user_query": "q1", "created_at": "t1"}
        
        mock_item2 = MagicMock()
        mock_item2.key = "conv2"
        mock_item2.value = {"user_query": "q2", "created_at": "t2"}
        
        mock_store.asearch.return_value = [mock_item1, mock_item2]
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                results = await list_conversations("user1", limit=10)
                
                assert len(results) == 2
                assert results[0]["conversation_id"] == "conv1"
                assert results[0]["user_query"] == "q1"
                assert results[1]["conversation_id"] == "conv2"
                mock_store.asearch.assert_awaited_once_with(("user1", "conversations"), limit=10)

    @pytest.mark.asyncio
    async def test_save_in_progress_conversation(self):
        mock_store = MagicMock()
        mock_store.setup = AsyncMock()
        mock_conn = MagicMock()
        mock_conn.close = AsyncMock()
        mock_store.conn = mock_conn
        
        mock_existing = MagicMock()
        mock_existing.value = {"thinking_state": "ts", "created_at": "old_date"}
        mock_store.aget = AsyncMock(return_value=mock_existing)
        mock_store.aput = AsyncMock()
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                await save_in_progress_conversation("u1", "c1", "query")
                
                mock_store.aput.assert_awaited_once()
                val = mock_store.aput.call_args.kwargs["value"]
                assert val["thinking_state"] == "ts"
                assert val["created_at"] == "old_date"
                assert val["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_update_conversation_status(self):
        mock_store = MagicMock()
        mock_store.setup = AsyncMock()
        mock_conn = MagicMock()
        mock_conn.close = AsyncMock()
        mock_store.conn = mock_conn
        
        mock_existing = MagicMock()
        mock_existing.value = {"status": "in_progress"}
        mock_store.aget = AsyncMock(return_value=mock_existing)
        mock_store.aput = AsyncMock()
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                await update_conversation_status("u1", "c1", "complete", "phase1", "rep")
                
                mock_store.aput.assert_awaited_once()
                val = mock_store.aput.call_args.kwargs["value"]
                assert val["status"] == "complete"
                assert val["phase"] == "phase1"
                assert val["report_content"] == "rep"
                
                # Test not found
                mock_store.aget.return_value = None
                mock_store.aput.reset_mock()
                await update_conversation_status("u1", "c2", "complete")
                mock_store.aput.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_thinking_state(self):
        mock_store = MagicMock()
        mock_store.setup = AsyncMock()
        mock_conn = MagicMock()
        mock_conn.close = AsyncMock()
        mock_store.conn = mock_conn
        
        mock_existing = MagicMock()
        mock_existing.value = {"thinking_state": None}
        mock_store.aget = AsyncMock(return_value=mock_existing)
        mock_store.aput = AsyncMock()
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                res = await update_thinking_state("u1", "c1", {"agent": "sup"})
                
                assert res is True
                mock_store.aput.assert_awaited_once()
                val = mock_store.aput.call_args[0][2] # args[2] is value for positional or apur(namespace, key, data)
                # wait, in store.py: await store.aput(namespace, conversation_id, data)
                
                # Test not found
                mock_store.aget.return_value = None
                res2 = await update_thinking_state("u1", "c2", {})
                assert res2 is False
