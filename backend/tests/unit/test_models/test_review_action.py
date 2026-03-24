"""Unit tests for ReviewAction model."""

import pytest
from pydantic import ValidationError

from app.models.schemas import ReviewAction


class TestReviewAction:
    """Test suite for ReviewAction model."""
    
    def test_valid_approve_action(self):
        """Test valid approve action without feedback."""
        action = ReviewAction(action="approve")
        
        assert action.action == "approve"
        assert action.feedback is None
    
    def test_valid_refine_action_with_feedback(self):
        """Test valid refine action with feedback."""
        feedback_text = "Please add more details about recent developments"
        action = ReviewAction(action="refine", feedback=feedback_text)
        
        assert action.action == "refine"
        assert action.feedback == feedback_text
    
    def test_valid_re_research_action_with_feedback(self):
        """Test valid re_research action with feedback."""
        feedback_text = "Research more about climate change impacts"
        action = ReviewAction(action="re_research", feedback=feedback_text)
        
        assert action.action == "re_research"
        assert action.feedback == feedback_text
    
    def test_invalid_action_raises_error(self):
        """Test invalid action value raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            ReviewAction(action="invalid_action")
        
        assert "Action must be one of" in str(exc.value)
    
    def test_missing_action_raises_error(self):
        """Test missing action field raises ValidationError."""
        with pytest.raises(ValidationError) as exc:
            ReviewAction()
        
        assert "action" in str(exc.value).lower()
    
    def test_action_with_empty_feedback(self):
        """Test action with empty string feedback is accepted."""
        action = ReviewAction(action="refine", feedback="")
        
        assert action.action == "refine"
        assert action.feedback == ""
    
    def test_model_dump_serialization(self):
        """Test ReviewAction serialization to dict."""
        action = ReviewAction(action="refine", feedback="Test feedback")
        
        result = action.model_dump()
        
        assert result["action"] == "refine"
        assert result["feedback"] == "Test feedback"
    
    def test_model_dump_json_serialization(self):
        """Test ReviewAction JSON serialization."""
        action = ReviewAction(action="approve")
        
        json_str = action.model_dump_json()
        
        assert "approve" in json_str
        assert isinstance(json_str, str)
