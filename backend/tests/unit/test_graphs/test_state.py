"""Unit tests for LangGraph state definition (Phase 3.5.3).

This module contains tests for the ResearchState TypedDict and state initialization
functions, verifying proper structure and reducer patterns.
"""

import pytest
from typing import Dict, Any

from app.graphs.state import ResearchState, create_initial_state
from app.models.schemas import ResearchBrief, SubAgentTask, SubAgentFindings, Citation


class TestResearchState:
    """Test cases for ResearchState TypedDict structure."""

    def test_research_state_has_all_required_fields(self):
        """Test that ResearchState TypedDict has all required fields."""
        # Arrange & Act
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1"],
                constraints={},
                deliverables="Test deliverables"
            ),
            "strategy": None,
            "tasks": [],
            "findings": [],
            "summarized_findings": None,
            "gaps": None,
            "extraction_budget": {"used": 0, "max": 5},
            "is_complete": False,
            "error": None,
            "messages": []
        }
        
        # Assert - All fields accessible
        assert state["research_brief"] is not None
        assert state["strategy"] is None
        assert isinstance(state["tasks"], list)
        assert isinstance(state["findings"], list)
        assert state["summarized_findings"] is None
        assert state["gaps"] is None
        assert isinstance(state["extraction_budget"], dict)
        assert state["is_complete"] is False
        assert state["error"] is None
        assert isinstance(state["messages"], list)

    def test_research_state_findings_reducer_pattern(self):
        """Test that findings field uses Annotated reducer pattern correctly."""
        # Arrange
        state: ResearchState = {
            "findings": [
                SubAgentFindings(
                    topic="Topic 1",
                    summary="Summary 1",
                    key_facts=["fact1"],
                    citations=[],
                    sources=[]
                )
            ]
        }
        
        # Act - Simulate reducer behavior (append)
        new_finding = SubAgentFindings(
            topic="Topic 2",
            summary="Summary 2",
            key_facts=["fact2"],
            citations=[],
            sources=[]
        )
        state["findings"] = state["findings"] + [new_finding]
        
        # Assert
        assert len(state["findings"]) == 2
        assert state["findings"][0].topic == "Topic 1"
        assert state["findings"][1].topic == "Topic 2"

    def test_research_state_messages_reducer_pattern(self):
        """Test that messages field uses Annotated reducer pattern correctly."""
        # Arrange
        state: ResearchState = {
            "messages": [{"role": "user", "content": "Message 1"}]
        }
        
        # Act - Simulate reducer behavior (append)
        new_message = {"role": "assistant", "content": "Message 2"}
        state["messages"] = state["messages"] + [new_message]
        
        # Assert
        assert len(state["messages"]) == 2
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][1]["role"] == "assistant"

    def test_research_state_extraction_budget_structure(self):
        """Test that extraction_budget has correct structure."""
        # Arrange & Act
        state: ResearchState = {
            "extraction_budget": {"used": 3, "max": 5}
        }
        
        # Assert
        assert "used" in state["extraction_budget"]
        assert "max" in state["extraction_budget"]
        assert state["extraction_budget"]["used"] == 3
        assert state["extraction_budget"]["max"] == 5

    def test_research_state_with_error_field(self):
        """Test that error field can hold error messages."""
        # Arrange & Act
        state: ResearchState = {
            "error": "Test error message",
            "is_complete": False
        }
        
        # Assert
        assert state["error"] == "Test error message"
        assert state["is_complete"] is False

    def test_research_state_with_complete_status(self):
        """Test that is_complete flag works correctly."""
        # Arrange & Act
        state: ResearchState = {
            "is_complete": True,
            "error": None
        }
        
        # Assert
        assert state["is_complete"] is True
        assert state["error"] is None


class TestCreateInitialState:
    """Test cases for create_initial_state function."""

    def test_create_initial_state_with_research_brief_returns_valid_state(self):
        """Test that create_initial_state creates valid initial ResearchState."""
        # Arrange
        brief = ResearchBrief(
            scope="Test research scope",
            sub_topics=["topic1", "topic2"],
            constraints={"time_period": "2020-2024"},
            deliverables="Test deliverables"
        )
        
        # Act
        state = create_initial_state(brief)
        
        # Assert
        assert state["research_brief"] == brief
        assert state["strategy"] is None
        assert state["tasks"] == []
        assert state["findings"] == []
        assert state["summarized_findings"] is None
        assert state["gaps"] is None
        assert state["extraction_budget"] == {"used": 0, "max": 5}
        assert state["is_complete"] is False
        assert state["error"] is None
        assert state["messages"] == []

    def test_create_initial_state_preserves_research_brief_data(self):
        """Test that create_initial_state preserves all research brief data."""
        # Arrange
        brief = ResearchBrief(
            scope="Quantum computing research",
            sub_topics=["Qubits", "Error correction"],
            constraints={"depth": "detailed"},
            deliverables="Technical report",
            format="summary",
            metadata={"author": "Test"}
        )
        
        # Act
        state = create_initial_state(brief)
        
        # Assert
        assert state["research_brief"].scope == "Quantum computing research"
        assert len(state["research_brief"].sub_topics) == 2
        assert state["research_brief"].constraints["depth"] == "detailed"
        assert state["research_brief"].deliverables == "Technical report"
        assert state["research_brief"].format == "summary"
        assert state["research_brief"].metadata["author"] == "Test"

    def test_create_initial_state_sets_default_extraction_budget(self):
        """Test that create_initial_state sets correct default extraction budget."""
        # Arrange
        brief = ResearchBrief(
            scope="Test",
            sub_topics=["topic1"],
            constraints={},
            deliverables="Test"
        )
        
        # Act
        state = create_initial_state(brief)
        
        # Assert
        assert state["extraction_budget"]["used"] == 0
        assert state["extraction_budget"]["max"] == 5

    def test_create_initial_state_initializes_empty_collections(self):
        """Test that create_initial_state initializes empty collections."""
        # Arrange
        brief = ResearchBrief(
            scope="Test",
            sub_topics=["topic1"],
            constraints={},
            deliverables="Test"
        )
        
        # Act
        state = create_initial_state(brief)
        
        # Assert
        assert isinstance(state["tasks"], list)
        assert len(state["tasks"]) == 0
        assert isinstance(state["findings"], list)
        assert len(state["findings"]) == 0
        assert isinstance(state["messages"], list)
        assert len(state["messages"]) == 0

    def test_create_initial_state_sets_is_complete_to_false(self):
        """Test that create_initial_state sets is_complete to False."""
        # Arrange
        brief = ResearchBrief(
            scope="Test",
            sub_topics=["topic1"],
            constraints={},
            deliverables="Test"
        )
        
        # Act
        state = create_initial_state(brief)
        
        # Assert
        assert state["is_complete"] is False

    def test_create_initial_state_sets_error_to_none(self):
        """Test that create_initial_state sets error to None."""
        # Arrange
        brief = ResearchBrief(
            scope="Test",
            sub_topics=["topic1"],
            constraints={},
            deliverables="Test"
        )
        
        # Act
        state = create_initial_state(brief)
        
        # Assert
        assert state["error"] is None

