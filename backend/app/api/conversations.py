"""API endpoints for conversation history."""

from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
from app.persistence.store import get_conversation, list_conversations

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationSummary(BaseModel):
    conversation_id: str
    user_query: str
    created_at: str


class ConversationDetail(BaseModel):
    conversation_id: str
    user_query: str
    report_content: str
    findings_count: int
    created_at: str


@router.get("/{user_id}", response_model=List[ConversationSummary])
async def list_user_conversations(user_id: str, limit: int = 50):
    """List all conversations for a user."""
    conversations = await list_conversations(user_id, limit=limit)
    return [ConversationSummary(**conv) for conv in conversations]


@router.get("/{user_id}/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(user_id: str, conversation_id: str):
    """Retrieve a specific conversation."""
    conversation = await get_conversation(user_id, conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetail(
        conversation_id=conversation["conversation_id"],
        user_query=conversation["user_query"],
        report_content=conversation["report_content"],
        findings_count=len(conversation["findings"]),
        created_at=conversation["created_at"]
    )
