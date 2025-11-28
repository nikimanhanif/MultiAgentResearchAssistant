"""Integration tests for LangGraph persistence."""

import pytest
import uuid
from app.persistence.checkpointer import initialize_checkpointer, shutdown_checkpointer
from app.persistence.store import (
    initialize_store,
    shutdown_store,
    save_conversation,
    get_conversation,
    list_conversations,
)
from app.models.schemas import ResearchBrief, Finding, Citation, SourceType


class TestCheckpointerInitialization:
    """Test checkpointer initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_checkpointer_initializes_successfully(self):
        """Test that checkpointer initializes and creates database."""
        checkpointer = await initialize_checkpointer()
        assert checkpointer is not None
        assert checkpointer.is_setup is True
        await shutdown_checkpointer()

    @pytest.mark.asyncio
    async def test_checkpointer_setup_is_idempotent(self):
        """Test that calling setup multiple times doesn't cause errors."""
        checkpointer = await initialize_checkpointer()
        await checkpointer.setup()
        await checkpointer.setup()
        assert checkpointer.is_setup is True
        await shutdown_checkpointer()


class TestStoreOperations:
    """Test store save and retrieve operations."""

    @pytest.mark.asyncio
    async def test_store_initializes_successfully(self):
        """Test that store initializes and creates database."""
        store = await initialize_store()
        assert store is not None
        assert store.is_setup is True
        await shutdown_store()

    @pytest.mark.asyncio
    async def test_save_and_retrieve_conversation(self):
        """Test saving and retrieving conversations from Store."""
        await initialize_store()

        user_id = "test_user"
        conversation_id = str(uuid.uuid4())
        
        brief = ResearchBrief(
            scope="Test research on AI",
            sub_topics=["machine learning", "deep learning"],
            constraints={},
            deliverables="Technical report"
        )
        
        citation = Citation(
            source="Nature",
            url="https://example.com",
            title="Test Article",
            credibility_score=0.9,
            source_type=SourceType.PEER_REVIEWED
        )
        
        findings = [
            Finding(
                claim="AI models improve accuracy",
                topic="machine learning",
                citation=citation,
                credibility_score=0.9
            )
        ]

        await save_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            user_query="Tell me about AI",
            research_brief=brief,
            findings=findings,
            report_content="# Research Report\n\nFindings on AI..."
        )

        retrieved = await get_conversation(user_id, conversation_id)
        assert retrieved is not None
        assert retrieved["conversation_id"] == conversation_id
        assert retrieved["user_query"] == "Tell me about AI"
        assert len(retrieved["findings"]) == 1
        
        await shutdown_store()

    @pytest.mark.asyncio
    async def test_get_nonexistent_conversation_returns_none(self):
        """Test retrieving non-existent conversation returns None."""
        await initialize_store()

        retrieved = await get_conversation("test_user", "nonexistent_id")
        assert retrieved is None
        
        await shutdown_store()

    @pytest.mark.asyncio
    async def test_list_conversations_returns_metadata(self):
        """Test listing conversations returns metadata."""
        await initialize_store()

        user_id = "test_user_list"
        conv_id_1 = str(uuid.uuid4())
        conv_id_2 = str(uuid.uuid4())
        
        brief = ResearchBrief(
            scope="Test",
            sub_topics=["topic1"],
            constraints={},
            deliverables="Report"
        )
        
        citation = Citation(
            source="Test Source",
            credibility_score=0.8,
            source_type=SourceType.ACADEMIC
        )
        
        finding = Finding(
            claim="Test claim",
            topic="topic1",
            citation=citation,
            credibility_score=0.8
        )

        for conv_id, query in [(conv_id_1, "Query 1"), (conv_id_2, "Query 2")]:
            await save_conversation(
                user_id=user_id,
                conversation_id=conv_id,
                user_query=query,
                research_brief=brief,
                findings=[finding],
                report_content="# Report"
            )

        conversations = await list_conversations(user_id, limit=50)
        assert len(conversations) >= 2
        
        conversation_ids = [conv["conversation_id"] for conv in conversations]
        assert conv_id_1 in conversation_ids
        assert conv_id_2 in conversation_ids
        
        await shutdown_store()

    @pytest.mark.asyncio
    async def test_list_conversations_respects_limit(self):
        """Test that list_conversations respects the limit parameter."""
        await initialize_store()

        user_id = "test_user_limit"
        
        brief = ResearchBrief(
            scope="Test",
            sub_topics=["topic"],
            constraints={},
            deliverables="Report"
        )
        
        citation = Citation(
            source="Source",
            credibility_score=0.8,
            source_type=SourceType.ACADEMIC
        )
        
        finding = Finding(
            claim="Claim",
            topic="topic",
            citation=citation,
            credibility_score=0.8
        )

        for i in range(5):
            await save_conversation(
                user_id=user_id,
                conversation_id=str(uuid.uuid4()),
                user_query=f"Query {i}",
                research_brief=brief,
                findings=[finding],
                report_content="# Report"
            )

        conversations = await list_conversations(user_id, limit=3)
        assert len(conversations) <= 3
        
        await shutdown_store()
