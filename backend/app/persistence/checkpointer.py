"""
LangGraph checkpointer for research graph state persistence.

Provides initialization, shutdown, and retrieval of the AsyncSqliteSaver
used for checkpointing the research graph state.
"""

from pathlib import Path
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHECKPOINT_DB_PATH = Path(__file__).parent.parent.parent / "checkpoints.db"

_checkpointer: AsyncSqliteSaver | None = None


async def initialize_checkpointer() -> AsyncSqliteSaver:
    """
    Initialize the checkpointer with SQLite backend.
    
    Called during FastAPI lifespan startup.
    
    Returns:
        AsyncSqliteSaver: Configured checkpointer instance.
    """
    global _checkpointer

    conn_string = str(CHECKPOINT_DB_PATH)
    
    import aiosqlite
    conn = await aiosqlite.connect(conn_string, isolation_level=None)
    _checkpointer = AsyncSqliteSaver(conn)
    await _checkpointer.setup()

    return _checkpointer


async def shutdown_checkpointer() -> None:
    """
    Cleanup checkpointer resources.
    
    Called during FastAPI lifespan shutdown.
    """
    global _checkpointer
    if _checkpointer and _checkpointer.conn:
        await _checkpointer.conn.close()
    _checkpointer = None


def get_checkpointer() -> AsyncSqliteSaver:
    """
    Get the initialized checkpointer instance.
    
    Used when compiling the research graph.
    
    Returns:
        AsyncSqliteSaver: The global checkpointer instance.
        
    Raises:
        RuntimeError: If checkpointer is not initialized.
    """
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer not initialized. Call initialize_checkpointer() first."
        )
    return _checkpointer
