"""Unit tests for checkpointer persistence module."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.persistence.checkpointer import (
    initialize_checkpointer,
    shutdown_checkpointer,
    get_checkpointer,
)

# Reset global state before each test
@pytest.fixture(autouse=True)
async def reset_checkpointer():
    await shutdown_checkpointer()
    yield
    await shutdown_checkpointer()

class TestCheckpointer:
    """Test suite for checkpointer management."""

    @pytest.mark.asyncio
    async def test_initialize_checkpointer_success(self):
        """Test successful initialization of checkpointer."""
        with patch("aiosqlite.connect", new_callable=AsyncMock) as mock_connect:
            with patch("app.persistence.checkpointer.AsyncSqliteSaver") as mock_saver_cls:
                mock_conn = AsyncMock()
                mock_connect.return_value = mock_conn
                
                mock_saver = AsyncMock()
                mock_saver_cls.return_value = mock_saver
                
                result = await initialize_checkpointer()
                
                assert result == mock_saver
                mock_connect.assert_called_once()
                mock_saver_cls.assert_called_once_with(mock_conn)
                mock_saver.setup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_checkpointer_closes_connection(self):
        """Test that shutdown closes the database connection."""
        mock_conn = AsyncMock()
        mock_saver = AsyncMock()
        mock_saver.conn = mock_conn
        
        with patch("aiosqlite.connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = mock_conn
            with patch("app.persistence.checkpointer.AsyncSqliteSaver", return_value=mock_saver):
                await initialize_checkpointer()
                
                await shutdown_checkpointer()
                
                mock_conn.close.assert_awaited_once()
                
                # Verify it's cleared
                with pytest.raises(RuntimeError, match="Checkpointer not initialized"):
                    get_checkpointer()

    @pytest.mark.asyncio
    async def test_shutdown_checkpointer_handles_none(self):
        """Test that shutdown handles uninitialized checkpointer gracefully."""
        await shutdown_checkpointer()

    @pytest.mark.asyncio
    async def test_get_checkpointer_returns_singleton(self):
        """Test that get_checkpointer returns the initialized instance."""
        with patch("aiosqlite.connect", new_callable=AsyncMock):
            with patch("app.persistence.checkpointer.AsyncSqliteSaver") as mock_saver_cls:
                mock_saver = AsyncMock()
                mock_saver_cls.return_value = mock_saver
                
                await initialize_checkpointer()
                
                result1 = get_checkpointer()
                result2 = get_checkpointer()
                
                assert result1 == mock_saver
                assert result1 is result2

    def test_get_checkpointer_raises_if_not_initialized(self):
        """Test that get_checkpointer raises RuntimeError if not initialized."""
        with pytest.raises(RuntimeError, match="Checkpointer not initialized"):
            get_checkpointer()
