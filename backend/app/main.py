from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.persistence import (
    initialize_checkpointer,
    shutdown_checkpointer,
    initialize_store,
    shutdown_store,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle.

    Initializes and shuts down persistence layers (checkpointer and store)
    when the application starts and stops.
    """
    await initialize_checkpointer()
    await initialize_store()
    yield
    await shutdown_checkpointer()
    await shutdown_store()


app = FastAPI(
    title="Multi-Agent Research Assistant API",
    description="FastAPI backend for multi-agent AI research assistant",
    version="0.1.0",
    lifespan=lifespan,
)

from app.api import conversations

app.include_router(conversations.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Multi-Agent Research Assistant API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

