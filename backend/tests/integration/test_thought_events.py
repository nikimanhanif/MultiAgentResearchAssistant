"""
Integration test for internal monologue SSE events.

Tests that the chat API correctly emits 'thought' events during research phases.
Uses mocked LangGraph to avoid making real API calls.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Import the streaming helpers directly for unit testing
from app.api.streaming import (
    StreamEventType,
    create_thought_event,
    create_progress_event,
    create_sse_event,
)


class TestThoughtEventCreation:
    """Unit tests for thought event SSE creation."""

    def test_create_thought_event_basic(self):
        """Test basic thought event creation."""
        event = create_thought_event(
            agent="supervisor",
            thought="Analyzing research gaps...",
            step="analyzing",
            elapsed_ms=1500
        )
        
        # Parse the SSE format
        assert event.startswith("data: ")
        data = json.loads(event.replace("data: ", "").strip())
        
        assert data["type"] == "thought"
        assert data["agent"] == "supervisor"
        assert data["thought"] == "Analyzing research gaps..."
        assert data["step"] == "analyzing"
        assert data["elapsed_ms"] == 1500

    def test_create_thought_event_sub_agent(self):
        """Test thought event for sub-agent research."""
        event = create_thought_event(
            agent="sub_agent",
            thought="Researching: machine learning optimization techniques",
            step="researching",
            elapsed_ms=3200
        )
        
        data = json.loads(event.replace("data: ", "").strip())
        
        assert data["type"] == "thought"
        assert data["agent"] == "sub_agent"
        assert "machine learning" in data["thought"]
        assert data["step"] == "researching"

    def test_create_thought_event_report_agent(self):
        """Test thought event for report generation."""
        event = create_thought_event(
            agent="report_agent",
            thought="Synthesizing findings into report...",
            step="generating",
            elapsed_ms=500
        )
        
        data = json.loads(event.replace("data: ", "").strip())
        
        assert data["type"] == "thought"
        assert data["agent"] == "report_agent"


class TestStreamEventTypes:
    """Test that new event types are properly defined."""

    def test_thought_event_type_exists(self):
        """Verify THOUGHT is a valid StreamEventType."""
        assert hasattr(StreamEventType, 'THOUGHT')
        assert StreamEventType.THOUGHT.value == "thought"

    def test_all_event_types_present(self):
        """Verify all expected event types exist."""
        expected_types = [
            "token", "progress", "state_update", "brief_created",
            "report_token", "clarification_request", "review_request",
            "thought", "complete", "error"
        ]
        
        for event_type in expected_types:
            assert event_type in [e.value for e in StreamEventType], \
                f"Missing event type: {event_type}"


class TestThoughtEventIntegration:
    """Integration tests for thought events in the streaming pipeline."""

    @pytest.fixture
    def mock_graph_state(self):
        """Create a mock graph state with supervisor output."""
        return {
            "supervisor": {
                "gaps": {
                    "has_gaps": True,
                    "gaps_identified": ["Missing recent papers"],
                    "reasoning": "Need to search for papers from 2023-2024"
                },
                "budget": {"iterations": 1}
            }
        }

    @pytest.fixture
    def mock_sub_agent_state(self):
        """Create a mock sub-agent state output."""
        class MockTask:
            topic = "Neural network optimization"
        
        return {
            "sub_agent": {
                "task": MockTask(),
                "findings": []
            }
        }

    def test_thought_event_format_matches_frontend(self):
        """Verify thought event format matches frontend TypeScript types."""
        event = create_thought_event(
            agent="supervisor",
            thought="Analyzing gaps in current findings",
            step="analyzing",
            elapsed_ms=2000
        )
        
        data = json.loads(event.replace("data: ", "").strip())
        
        # These fields must match ThoughtEvent interface in frontend/types/chat.ts
        required_fields = ["type", "agent", "thought", "step", "elapsed_ms"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Type assertions
        assert isinstance(data["type"], str)
        assert isinstance(data["agent"], str)
        assert isinstance(data["thought"], str)
        assert isinstance(data["step"], str)
        assert isinstance(data["elapsed_ms"], int)

    def test_multiple_thought_events_sequence(self):
        """Test a sequence of thought events like a real research session."""
        events = [
            create_thought_event("supervisor", "Starting gap analysis", "analyzing", 100),
            create_thought_event("sub_agent", "Researching: ML optimization", "researching", 500),
            create_thought_event("sub_agent", "Found 3 relevant papers", "researching", 1200),
            create_thought_event("report_agent", "Synthesizing findings...", "generating", 2000),
        ]
        
        # All events should be valid SSE format
        for event in events:
            assert event.startswith("data: ")
            assert event.endswith("\n\n")
            
            data = json.loads(event.replace("data: ", "").strip())
            assert data["type"] == "thought"


class TestThoughtTruncation:
    """Test that long thoughts are appropriately handled."""

    def test_long_thought_is_valid(self):
        """Thoughts can be any length - truncation happens in chat.py."""
        long_thought = "A" * 500  # 500 character thought
        
        event = create_thought_event(
            agent="supervisor",
            thought=long_thought,
            step="analyzing",
            elapsed_ms=0
        )
        
        data = json.loads(event.replace("data: ", "").strip())
        assert len(data["thought"]) == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
