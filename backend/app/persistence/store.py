"""
LangGraph Store for long-term conversation persistence.

Provides initialization, shutdown, and CRUD operations for the AsyncSqliteStore,
used to persist both in-progress and completed conversations.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from langgraph.store.sqlite.aio import AsyncSqliteStore
from app.models.schemas import ResearchBrief, Finding

STORE_DB_PATH = Path(__file__).parent.parent.parent / "conversations.db"

_store: AsyncSqliteStore | None = None

# Conversation status types
ConversationStatus = Literal["in_progress", "waiting_review", "complete"]


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


async def save_in_progress_conversation(
    user_id: str,
    conversation_id: str,
    user_query: str,
    phase: str = "scoping",
) -> None:
    """
    Save or update an in-progress conversation.
    
    Called at the start of a new research session to track it immediately.
    
    Args:
        user_id: User identifier.
        conversation_id: Unique conversation/thread UUID.
        user_query: Original user query.
        phase: Current research phase (scoping, researching, report, review).
    """
    store = get_store()
    namespace = (user_id, "conversations")
    
    # Check if conversation already exists
    existing = await store.aget(namespace, conversation_id)
    
    conversation_data = {
        "conversation_id": conversation_id,
        "user_query": user_query,
        "status": "in_progress",
        "phase": phase,
        "research_brief": None,
        "findings": [],
        "report_content": "",
        "thinking_state": existing.value.get("thinking_state") if existing else None,
        "created_at": existing.value.get("created_at") if existing else datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    
    await store.aput(
        namespace=namespace,
        key=conversation_id,
        value=conversation_data,
    )


async def update_conversation_status(
    user_id: str,
    conversation_id: str,
    status: ConversationStatus,
    phase: Optional[str] = None,
    report_content: Optional[str] = None,
) -> None:
    """
    Update conversation status and optional fields.
    
    Args:
        user_id: User identifier.
        conversation_id: Conversation UUID.
        status: New status (in_progress, waiting_review, complete).
        phase: Optional phase update.
        report_content: Optional report content update.
    """
    store = get_store()
    namespace = (user_id, "conversations")
    
    existing = await store.aget(namespace, conversation_id)
    if not existing:
        return  # Conversation doesn't exist, nothing to update
    
    data = existing.value.copy()
    data["status"] = status
    data["updated_at"] = datetime.now().isoformat()
    
    if phase is not None:
        data["phase"] = phase
    if report_content is not None:
        data["report_content"] = report_content
    
    await store.aput(
        namespace=namespace,
        key=conversation_id,
        value=data,
    )


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
    
    This updates an existing in-progress conversation to complete status,
    or creates a new complete conversation if it doesn't exist.
    Preserves existing thinking_state if present.
    
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
    
    # Check if conversation already exists (in-progress)
    existing = await store.aget(namespace, conversation_id)

    conversation_data = {
        "conversation_id": conversation_id,
        "user_query": user_query,
        "status": "complete",
        "phase": "complete",
        "research_brief": research_brief.model_dump(mode='json'),
        "findings": [f.model_dump(mode='json') for f in findings],
        "report_content": report_content,
        "thinking_state": existing.value.get("thinking_state") if existing else None,
        "created_at": existing.value.get("created_at") if existing else datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
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
        List[Dict[str, Any]]: List of conversation metadata dicts including status.
    """
    store = get_store()
    namespace = (user_id, "conversations")

    results = await store.asearch(namespace, limit=limit)

    return [
        {
            "conversation_id": item.key,
            "user_query": item.value.get("user_query"),
            "created_at": item.value.get("created_at"),
            "status": item.value.get("status", "complete"),  # Default to complete for legacy data
            "phase": item.value.get("phase"),
        }
        for item in results
    ]


async def update_thinking_state(
    user_id: str,
    conversation_id: str,
    thinking_state: Dict[str, Any]
) -> bool:
    """
    Update just the thinking_state field of a conversation.
    
    Called by frontend to persist the thinking block state.
    
    Args:
        user_id: User identifier.
        conversation_id: Conversation UUID.
        thinking_state: The thinking state object from frontend.
        
    Returns:
        bool: True if update succeeded, False if conversation not found.
    """
    store = get_store()
    namespace = (user_id, "conversations")
    
    existing = await store.aget(namespace, conversation_id)
    if not existing:
        return False
    
    data = existing.value.copy()
    data["thinking_state"] = thinking_state
    data["updated_at"] = datetime.now().isoformat()
    
    await store.aput(namespace, conversation_id, data)
    return True
