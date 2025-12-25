"""
API module containing all API endpoints and routers.

This module includes:
- Chat API: Unified streaming endpoint for the research pipeline.
- Conversations API: History management for research sessions.
"""

from app.api.routes import api_router
from app.api import chat, conversations

__all__ = ["api_router", "chat", "conversations"]
