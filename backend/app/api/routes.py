"""API Routes."""

from fastapi import APIRouter
from app.api import conversations, chat

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(conversations.router)
api_router.include_router(chat.router)

__all__ = ["api_router"]
