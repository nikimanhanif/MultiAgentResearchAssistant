"""
LangGraph Store for long-term conversation persistence.

Provides initialization, shutdown, and CRUD operations for the AsyncSqliteStore,
used to persist completed conversations and research results.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from langgraph.store.sqlite.aio import AsyncSqliteStore
from app.models.schemas import ResearchBrief, Finding

STORE_DB_PATH = Path(__file__).parent.parent.parent / "conversations.db"

_store: AsyncSqliteStore | None = None


async def initialize_store() -> AsyncSqliteStore:
    """
    Initialize the Store for long-term memory.
    
    Uses AsyncSqliteStore for persistent storage.
    
    Returns:
        AsyncSqliteStore: Configured store instance.
    """
    global _store

    conn_string = str(STORE_DB_PATH)
    
    import aiosqlite
    conn = await aiosqlite.connect(conn_string, isolation_level=None)
    _store = AsyncSqliteStore(conn)
    await _store.setup()

    return _store


async def shutdown_store() -> None:
    """Cleanup store resources."""
    global _store
    if _store and _store.conn:
        await _store.conn.close()
    _store = None


def get_store() -> AsyncSqliteStore:
    """
    Get the initialized store instance.
    
    Used when compiling the research graph and in API endpoints.
    
    Returns:
        AsyncSqliteStore: The global store instance.
        
    Raises:
        RuntimeError: If store is not initialized.
    """
    if _store is None:
        raise RuntimeError(
            "Store not initialized. Call initialize_store() first."
        )
    return _store


async def save_conversation(
    user_id: str,
    conversation_id: str,
    user_query: str,
    research_brief: ResearchBrief,
    findings: List[Finding],
    report_content: str
) -> None:
    """
    Save completed conversation to Store.
    
    Args:
        user_id: User identifier.
        conversation_id: Unique conversation UUID.
        user_query: Original user query.
        research_brief: Generated research brief.
        findings: List of research findings.
        report_content: Final markdown report.
    """
    store = get_store()
    namespace = (user_id, "conversations")

    conversation_data = {
        "conversation_id": conversation_id,
        "user_query": user_query,
        "research_brief": research_brief.model_dump(mode='json'),
        "findings": [f.model_dump(mode='json') for f in findings],
        "report_content": report_content,
        "created_at": datetime.now().isoformat(),
    }

    await store.aput(
        namespace=namespace,
        key=conversation_id,
        value=conversation_data,
    )


async def get_conversation(
    user_id: str,
    conversation_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific conversation from Store.
    
    Args:
        user_id: User identifier.
        conversation_id: Conversation UUID.
        
    Returns:
        Optional[Dict[str, Any]]: Conversation data dict or None if not found.
    """
    store = get_store()
    namespace = (user_id, "conversations")

    result = await store.aget(namespace, conversation_id)
    return result.value if result else None


async def list_conversations(
    user_id: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    List all conversations for a user.
    
    Args:
        user_id: User identifier.
        limit: Maximum number of conversations to return.
        
    Returns:
        List[Dict[str, Any]]: List of conversation metadata dicts.
    """
    store = get_store()
    namespace = (user_id, "conversations")

    results = await store.asearch(namespace, limit=limit)

    return [
        {
            "conversation_id": item.key,
            "user_query": item.value.get("user_query"),
            "created_at": item.value.get("created_at"),
        }
        for item in results
    ]
