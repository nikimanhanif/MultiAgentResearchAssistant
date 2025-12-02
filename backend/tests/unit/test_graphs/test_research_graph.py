"""Unit tests for research graph structure and routing logic."""

import pytest
from typing import List
from langgraph.types import Send
from langgraph.graph import START, END

from app.graphs.research_graph import route_from_supervisor, route_from_scope, build_research_graph
from app.graphs.state import ResearchState, create_initial_state
from app.models.schemas import ResearchBrief, ResearchTask, Finding, Citation, SourceType


class TestRouteFromSupervisor:
    """Tests for route_from_supervisor conditional edge function."""
    
    def test_route_from_supervisor_complete_flag_routes_to_report(self):
        """Test that is_complete flag routes to report_agent."""
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={"iterations": 1, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=True,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_supervisor(state)
        assert result == "report_agent"
    
    def test_route_tasks_max_iterations_routes_to_end(self):
        """Test that exceeding max_iterations routes to END."""
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[ResearchTask(task_id="task1", topic="test", query="test", priority=1)],
            completed_tasks=[],
            failed_tasks=[],
            budget={"iterations": 20, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_supervisor(state)
        assert result == "report_agent"
    
    def test_route_from_supervisor_max_sub_agents_routes_to_report(self):
        """Test that exceeding max_sub_agents routes to report_agent."""
        findings = [
            Finding(
                claim=f"claim {i}",
                citation=Citation(source=f"source {i}", url=f"http://example.com/{i}", title=f"title {i}"),
                topic="test",
                credibility_score=0.8
            )
            for i in range(20)
        ]
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=findings,
            task_history=[ResearchTask(task_id="task1", topic="test", query="test", priority=1)],
            completed_tasks=[],
            failed_tasks=[],
            budget={"iterations": 5, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_supervisor(state)
        assert result == "report_agent"
    
    def test_route_from_supervisor_no_pending_tasks_routes_to_report(self):
        """Test that no pending tasks routes to report_agent."""
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={"iterations": 1, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_supervisor(state)
        assert result == "report_agent"
    
    def test_route_from_supervisor_pending_tasks_creates_send_objects(self):
        """Test that pending tasks create Send objects for parallel execution."""
        task1 = ResearchTask(task_id="task1", topic="topic1", query="query1", priority=1)
        task2 = ResearchTask(task_id="task2", topic="topic2", query="query2", priority=1)
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[task1, task2],
            completed_tasks=[],
            failed_tasks=[],
            budget={"iterations": 1, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_supervisor(state)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(send, Send) for send in result)
    
    def test_route_from_supervisor_filters_completed_tasks(self):
        """Test that completed tasks are filtered out."""
        task1 = ResearchTask(task_id="task1", topic="topic1", query="query1", priority=1)
        task2 = ResearchTask(task_id="task2", topic="topic2", query="query2", priority=1)
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[task1, task2],
            completed_tasks=["task1"],
            failed_tasks=[],
            budget={"iterations": 1, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_supervisor(state)
        
        # Should only create Send for task2
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_route_from_supervisor_filters_failed_tasks(self):
        """Test that failed tasks are filtered out."""
        task1 = ResearchTask(task_id="task1", topic="topic1", query="query1", priority=1)
        task2 = ResearchTask(task_id="task2", topic="topic2", query="query2", priority=1)
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[task1, task2],
            completed_tasks=[],
            failed_tasks=["task1"],
            budget={"iterations": 1, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_supervisor(state)
        
        # Should only create Send for task2
        assert isinstance(result, list)
        assert len(result) == 1


class TestBuildResearchGraph:
    """Tests for build_research_graph function."""
    
    def test_build_research_graph_creates_compiled_graph(self):
        """Test that build_research_graph creates a compiled graph."""
        from unittest.mock import patch, MagicMock
        
        with patch("app.graphs.research_graph.get_checkpointer") as mock_get_cp:
            mock_cp = MagicMock()
            mock_get_cp.return_value = mock_cp
            
            graph = build_research_graph()
        
        assert graph is not None
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")
    
    def test_research_graph_has_required_nodes(self):
        """Test that graph has supervisor and sub_agent nodes."""
        from unittest.mock import patch, MagicMock
        
        with patch("app.graphs.research_graph.get_checkpointer") as mock_get_cp:
            mock_cp = MagicMock()
            mock_get_cp.return_value = mock_cp
            
            graph = build_research_graph()
        
        node_names = list(graph.nodes.keys())
        assert "scope" in node_names
        assert "supervisor" in node_names
        assert "sub_agent" in node_names
        assert "report_agent" in node_names
        assert "reviewer" in node_names

class TestRouteFromScope:
    """Tests for route_from_scope conditional edge function."""
    
    def test_route_from_scope_with_brief_routes_to_supervisor(self):
        """Test that existence of research_brief routes to supervisor."""
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={"iterations": 0, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_scope(state)
        assert result == "supervisor"

    def test_route_from_scope_without_brief_routes_to_end(self):
        """Test that missing research_brief routes to END."""
        state = ResearchState(
            research_brief=None,
            findings=[],
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={"iterations": 0, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_from_scope(state)
        assert result == END
