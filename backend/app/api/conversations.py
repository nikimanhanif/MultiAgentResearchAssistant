"""
Conversation History API.

Provides endpoints for listing and retrieving past research conversations
stored in the persistence layer, including in-progress conversations.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
from app.persistence import (
    get_checkpointer, 
    get_store, 
    get_conversation, 
    list_conversations,
    update_thinking_state
)
from app.graphs.research_graph import build_research_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationSummary(BaseModel):
    """Summary model for conversation listing."""
    conversation_id: str
    user_query: str
    created_at: str
    status: str = "complete"  # in_progress, waiting_review, complete
    phase: Optional[str] = None


class ConversationDetail(BaseModel):
    """Detailed model for a specific conversation."""
    conversation_id: str
    user_query: str
    report_content: str
    findings_count: int
    created_at: str
    status: str = "complete"
    phase: Optional[str] = None
    messages: List[Dict[str, str]] = []
    thinking_state: Optional[Dict[str, Any]] = None


class ConversationState(BaseModel):
    """Model for in-progress conversation state."""
    conversation_id: str
    status: str
    phase: Optional[str] = None
    report_content: Optional[str] = None
    has_pending_interrupt: bool = False
    interrupt_type: Optional[str] = None


class ThinkingStateUpdate(BaseModel):
    """Request model for updating thinking state from frontend."""
    thinking_state: Dict[str, Any]


@router.get("/{user_id}", response_model=List[ConversationSummary])
async def list_user_conversations(user_id: str, limit: int = 50):
    """
    List all conversations for a user.
    
    Args:
        user_id: The ID of the user.
        limit: Maximum number of conversations to return.
        
    Returns:
        List[ConversationSummary]: List of conversation summaries including status.
    """
    conversations = await list_conversations(user_id, limit=limit)
    return [ConversationSummary(**conv) for conv in conversations]


@router.get("/{user_id}/{conversation_id}", response_model=ConversationDetail)
async def get_conversation_detail(user_id: str, conversation_id: str):
    """
    Retrieve a specific conversation.
    
    Args:
        user_id: The ID of the user.
        conversation_id: The ID of the conversation.
        
    Returns:
        ConversationDetail: Detailed conversation information.
        
    Raises:
        HTTPException: If the conversation is not found.
    """
    conversation = await get_conversation(user_id, conversation_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Fetch messages from the graph state
    messages = []
    try:
        graph = build_research_graph()
        config = {"configurable": {"thread_id": conversation_id}}
        graph_state = await graph.aget_state(config)
        
        if graph_state and graph_state.values:
            graph_messages = graph_state.values.get("messages", [])
            for msg in graph_messages:
                # Handle different message types/formats
                role = "assistant"
                content = ""
                
                if isinstance(msg, dict):
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "")
                elif hasattr(msg, "type"):
                    # LangChain message objects
                    if msg.type == "human":
                        role = "user"
                    elif msg.type == "ai":
                        role = "assistant"
                    else:
                        role = msg.type
                    content = msg.content
                
                # Only include user and assistant messages with content
                if content and role in ["user", "assistant"]:
                    messages.append({
                        "role": role,
                        "content": content
                    })
            
            # Check for pending interrupts (e.g. clarification questions)
            # These are not yet in 'messages' state but should be displayed
            if graph_state.tasks:
                for task in graph_state.tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        for interrupt_data in task.interrupts:
                            interrupt_value = interrupt_data.value if hasattr(interrupt_data, 'value') else interrupt_data
                            
                            if isinstance(interrupt_value, dict) and interrupt_value.get("type") == "clarification_request":
                                questions = interrupt_value.get("questions", "")
                                if questions:
                                    # Check if this exact message is already at the end (deduplication)
                                    # (in case of race conditions or partial commits)
                                    is_duplicate = False
                                    if messages:
                                        last_msg = messages[-1]
                                        if last_msg["role"] == "assistant" and last_msg["content"] == questions:
                                            is_duplicate = True
                                    
                                    if not is_duplicate:
                                        messages.append({
                                            "role": "assistant",
                                            "content": questions
                                        })
    except Exception as e:
        # Don't fail the whole request if message loading fails, just log it
        logger.warning(f"Error loading messages for {conversation_id}: {e}")

    return ConversationDetail(
        conversation_id=conversation["conversation_id"],
        user_query=conversation["user_query"],
        report_content=conversation.get("report_content", ""),
        findings_count=len(conversation.get("findings", [])),
        created_at=conversation["created_at"],
        status=conversation.get("status", "complete"),
        phase=conversation.get("phase"),
        messages=messages,
        thinking_state=conversation.get("thinking_state")
    )


@router.get("/{user_id}/{conversation_id}/state", response_model=ConversationState)
async def get_conversation_state(user_id: str, conversation_id: str):
    """
    Get the state of an in-progress conversation.
    
    Checks both the store and checkpointer to determine current status
    and whether there are pending interrupts (e.g., HITL review).
    
    Args:
        user_id: The ID of the user.
        conversation_id: The conversation/thread ID.
        
    Returns:
        ConversationState: Current state including interrupt status.
    """
    # Get stored conversation data
    conversation = await get_conversation(user_id, conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    status = conversation.get("status", "complete")
    phase = conversation.get("phase")
    report_content = conversation.get("report_content")
    
    # Check checkpointer for pending interrupts
    has_pending_interrupt = False
    interrupt_type = None
    
    if status in ("in_progress", "waiting_review"):
        try:
            graph = build_research_graph()
            config = {"configurable": {"thread_id": conversation_id}}
            graph_state = await graph.aget_state(config)
            
            if graph_state and graph_state.tasks:
                for task in graph_state.tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        has_pending_interrupt = True
                        for intr in task.interrupts:
                            if hasattr(intr, 'value') and isinstance(intr.value, dict):
                                interrupt_type = intr.value.get("type")
                                # Get report from interrupt if available
                                if not report_content and intr.value.get("report"):
                                    report_content = intr.value.get("report")
                        break
        except Exception:
            logger.debug(f"Graph state not available for conversation {conversation_id}")
    
    return ConversationState(
        conversation_id=conversation_id,
        status=status,
        phase=phase,
        report_content=report_content,
        has_pending_interrupt=has_pending_interrupt,
        interrupt_type=interrupt_type
    )


@router.delete("/{user_id}/{conversation_id}")
async def delete_conversation(user_id: str, conversation_id: str):
    """
    Delete a specific conversation.
    
    Args:
        user_id: The ID of the user.
        conversation_id: The ID of the conversation to delete.
        
    Returns:
        dict: Success message.
        
    Raises:
        HTTPException: If the conversation is not found.
    """
    store = get_store()
    namespace = (user_id, "conversations")
    
    # Check if conversation exists
    result = await store.aget(namespace, conversation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Delete the conversation from store
    await store.adelete(namespace, conversation_id)
    
    # Delete the conversation from checkpointer
    try:
        checkpointer = get_checkpointer()
        await checkpointer.conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", 
            (conversation_id,)
        )
        await checkpointer.conn.execute(
            "DELETE FROM writes WHERE thread_id = ?", 
            (conversation_id,)
        )
    except Exception as e:
        logger.warning(
            f"Failed to purge checkpoints for {conversation_id}: {e}"
        )
    
    return {"message": "Conversation deleted successfully"}


@router.patch("/{user_id}/{conversation_id}/thinking")
async def update_conversation_thinking(
    user_id: str, 
    conversation_id: str, 
    body: ThinkingStateUpdate
):
    """
    Update thinking state for a conversation.
    
    Called by frontend to persist thinking block state so it survives
    conversation switching and page refreshes.
    
    Args:
        user_id: The ID of the user.
        conversation_id: The ID of the conversation.
        body: ThinkingStateUpdate containing the thinking_state object.
        
    Returns:
        dict: Success message.
        
    Raises:
        HTTPException: If the conversation is not found.
    """
    success = await update_thinking_state(
        user_id=user_id,
        conversation_id=conversation_id,
        thinking_state=body.thinking_state
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"success": True}

