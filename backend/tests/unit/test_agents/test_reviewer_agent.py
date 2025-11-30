"""Unit tests for reviewer_agent.py (Phase 6.2).

Tests cover:
- reviewer_node interrupt behavior
- create_new_task helper function
- All three routing paths (approve, refine, re_research)
"""

import pytest
from unittest.mock import Mock, patch
from langgraph.types import Command
from langgraph.graph import END

from app.agents.reviewer_agent import reviewer_node, create_new_task
from app.graphs.state import ResearchState
from app.models.schemas import ResearchBrief, ResearchTask


class TestReviewerNode:
    """Test suite for reviewer_node function."""
    
    def test_reviewer_node_approve_action(self):
        """Test reviewer_node returns END command for approve action."""
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1"],
                deliverables="Test deliverables"
            ),
            "findings": [],
            "task_history": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {"iterations": 0, "max_iterations": 20, "max_sub_agents": 20},
            "gaps": None,
            "is_complete": False,
            "error": None,
            "messages": [],
            "report_content": "# Test Report\n\nThis is a test report.",
            "reviewer_feedback": None
        }
        
        with patch("app.agents.reviewer_agent.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "approve"}
            
            result = reviewer_node(state)
            
            assert isinstance(result, Command)
            assert result.goto == END
            mock_interrupt.assert_called_once()
            
            call_kwargs = mock_interrupt.call_args.kwargs
            if 'value' in call_kwargs:
                call_value = call_kwargs['value']
            else:
                call_value = mock_interrupt.call_args[0][0]
            
            assert call_value["type"] == "review_request"
            assert call_value["report"] == state["report_content"]
    
    def test_reviewer_node_refine_action(self):
        """Test reviewer_node returns report_agent command for refine action with feedback."""
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1"],
                deliverables="Test deliverables"
            ),
            "findings": [],
            "task_history": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {"iterations": 0, "max_iterations": 20, "max_sub_agents": 20},
            "gaps": None,
            "is_complete": False,
            "error": None,
            "messages": [],
            "report_content": "# Test Report",
            "reviewer_feedback": None
        }
        
        feedback_text = "Please add more details about X"
        
        with patch("app.agents.reviewer_agent.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {
                "action": "refine",
                "feedback": feedback_text
            }
            
            result = reviewer_node(state)
            
            assert isinstance(result, Command)
            assert result.goto == "report_agent"
            assert result.update["reviewer_feedback"] == feedback_text
    
    def test_reviewer_node_re_research_action(self):
        """Test reviewer_node returns supervisor_agent command for re_research action."""
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1"],
                deliverables="Test deliverables"
            ),
            "findings": [],
            "task_history": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {"iterations": 0, "max_iterations": 20, "max_sub_agents": 20},
            "gaps": None,
            "is_complete": False,
            "error": None,
            "messages": [],
            "report_content": "# Test Report",
            "reviewer_feedback": None
        }
        
        feedback_text = "Research more about recent developments in Y"
        
        with patch("app.agents.reviewer_agent.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {
                "action": "re_research",
                "feedback": feedback_text
            }
            
            result = reviewer_node(state)
            
            assert isinstance(result, Command)
            assert result.goto == "supervisor"
            assert result.update["reviewer_feedback"] is None
            assert result.update["is_complete"] is False
            assert len(result.update["task_history"]) == 1
            
            new_task = result.update["task_history"][0]
            assert isinstance(new_task, ResearchTask)
            assert new_task.topic == "user_requested"
            assert new_task.query == feedback_text
            assert new_task.priority == 1
            assert new_task.requested_by == "reviewer"
    
    def test_reviewer_node_fallback_to_end(self):
        """Test reviewer_node falls back to END for invalid action."""
        state: ResearchState = {
            "research_brief": ResearchBrief(
                scope="Test scope",
                sub_topics=["topic1"],
                deliverables="Test deliverables"
            ),
            "findings": [],
            "task_history": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "budget": {"iterations": 0, "max_iterations": 20, "max_sub_agents": 20},
            "gaps": None,
            "is_complete": False,
            "error": None,
            "messages": [],
            "report_content": "# Test Report",
            "reviewer_feedback": None
        }
        
        with patch("app.agents.reviewer_agent.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "invalid_action"}
            
            result = reviewer_node(state)
            
            assert isinstance(result, Command)
            assert result.goto == END


class TestCreateNewTask:
    """Test suite for create_new_task helper function."""
    
    def test_create_new_task_returns_valid_task(self):
        """Test create_new_task returns a valid ResearchTask."""
        feedback = "Investigate the impact of climate change on coral reefs"
        
        task = create_new_task(feedback)
        
        assert isinstance(task, ResearchTask)
        assert task.topic == "user_requested"
        assert task.query == feedback
        assert task.priority == 1
        assert task.requested_by == "reviewer"
        assert task.task_id.startswith("task_reviewer_")
        assert len(task.task_id.split("_")[-1]) == 8
    
    def test_create_new_task_unique_ids(self):
        """Test create_new_task generates unique task IDs."""
        feedback = "Same feedback"
        
        task1 = create_new_task(feedback)
        task2 = create_new_task(feedback)
        
        assert task1.task_id != task2.task_id
        assert task1.query == task2.query
