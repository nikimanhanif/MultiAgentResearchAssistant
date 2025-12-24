"""
Persistence module for LangGraph checkpointing and storage.

This module exports functionality for:
- Checkpointer: Managing graph state checkpoints (SQLite).
- Store: Managing long-term conversation history (SQLite).
"""

from app.persistence.checkpointer import (
    initialize_checkpointer,
    shutdown_checkpointer,
    get_checkpointer,
)
from app.persistence.store import (
    initialize_store,
    shutdown_store,
    get_store,
    save_conversation,
    get_conversation,
    list_conversations,
    save_in_progress_conversation,
    update_conversation_status,
    ConversationStatus,
)

__all__ = [
    "initialize_checkpointer",
    "shutdown_checkpointer",
    "get_checkpointer",
    "initialize_store",
    "shutdown_store",
    "get_store",
    "save_conversation",
    "get_conversation",
    "list_conversations",
    "save_in_progress_conversation",
    "update_conversation_status",
    "ConversationStatus",
]

