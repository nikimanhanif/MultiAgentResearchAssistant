"""
API Routes Configuration.

Aggregates all API routers (chat, conversations) into a single main router
with the /api/v1 prefix.
"""

from fastapi import APIRouter
from app.api import conversations, chat, exports

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(conversations.router)
api_router.include_router(chat.router)
api_router.include_router(exports.router)

__all__ = ["api_router"]
