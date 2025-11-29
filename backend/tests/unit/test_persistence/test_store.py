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
                # Arrange
                mock_conn = AsyncMock()
                mock_connect.return_value = mock_conn
                
                mock_store = AsyncMock()
                mock_store_cls.return_value = mock_store
                
                # Act
                result = await initialize_store()
                
                # Assert
                assert result == mock_store
                mock_connect.assert_called_once()
                mock_store_cls.assert_called_once_with(mock_conn)
                mock_store.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_store_closes_connection(self):
        """Test that shutdown closes the database connection."""
        # Arrange
        mock_conn = AsyncMock()
        mock_store = AsyncMock()
        mock_store.conn = mock_conn
        
        with patch("aiosqlite.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_conn
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                # Act
                await shutdown_store()
                
                # Assert
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
        # Arrange
        mock_store = AsyncMock()
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                brief = ResearchBrief(
                    scope="Test",
                    sub_topics=["t1"],
                    constraints={},
                    deliverables="d1",
                    format=ReportFormat.SUMMARY
                )
                findings = [
                    Finding(
                        claim="c1",
                        citation=Citation(source="s1", url="u1"),
                        topic="t1",
                        credibility_score=0.9
                    )
                ]
                
                # Act
                await save_conversation(
                    user_id="user1",
                    conversation_id="conv1",
                    user_query="query",
                    research_brief=brief,
                    findings=findings,
                    report_content="report"
                )
                
                # Assert
                mock_store.aput.assert_awaited_once()
                call_args = mock_store.aput.call_args
                assert call_args.kwargs["namespace"] == ("user1", "conversations")
                assert call_args.kwargs["key"] == "conv1"
                assert call_args.kwargs["value"]["user_query"] == "query"
                assert call_args.kwargs["value"]["report_content"] == "report"

    @pytest.mark.asyncio
    async def test_get_conversation_calls_store_get(self):
        """Test that get_conversation calls store.aget."""
        # Arrange
        mock_store = AsyncMock()
        mock_item = MagicMock()
        mock_item.value = {"data": "test"}
        mock_store.aget.return_value = mock_item
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                # Act
                result = await get_conversation("user1", "conv1")
                
                # Assert
                assert result == {"data": "test"}
                mock_store.aget.assert_awaited_once_with(("user1", "conversations"), "conv1")

    @pytest.mark.asyncio
    async def test_get_conversation_returns_none_if_not_found(self):
        """Test that get_conversation returns None if store returns None."""
        # Arrange
        mock_store = AsyncMock()
        mock_store.aget.return_value = None
        
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.store.AsyncSqliteStore", return_value=mock_store):
                await initialize_store()
                
                # Act
                result = await get_conversation("user1", "conv1")
                
                # Assert
                assert result is None

    @pytest.mark.asyncio
    async def test_list_conversations_calls_store_search(self):
        """Test that list_conversations calls store.asearch and formats results."""
        # Arrange
        mock_store = AsyncMock()
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
                
                # Act
                results = await list_conversations("user1", limit=10)
                
                # Assert
                assert len(results) == 2
                assert results[0]["conversation_id"] == "conv1"
                assert results[0]["user_query"] == "q1"
                assert results[1]["conversation_id"] == "conv2"
                
                mock_store.asearch.assert_awaited_once_with(("user1", "conversations"), limit=10)
