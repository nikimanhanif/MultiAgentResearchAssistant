"""Unit tests for research graph structure and routing logic.

"""

import pytest
from typing import List
from langgraph.types import Send
from langgraph.graph import START, END

from app.graphs.research_graph import route_tasks, build_research_graph
from app.graphs.state import ResearchState, create_initial_state
from app.models.schemas import ResearchBrief, ResearchTask, Finding, Citation, SourceType


class TestRouteTasks:
    """Tests for route_tasks conditional edge function."""
    
    def test_route_tasks_complete_flag_routes_to_end(self):
        """Test that is_complete flag routes to END."""
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={"iterations": 1, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=True,  # Set complete flag
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_tasks(state)
        assert result == END
    
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
            budget={"iterations": 20, "max_iterations": 20, "max_sub_agents": 20},  # At limit
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_tasks(state)
        assert result == END
    
    def test_route_tasks_max_sub_agents_routes_to_end(self):
        """Test that exceeding max_sub_agents routes to END."""
        # Create 20 findings to hit the limit
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
            budget={"iterations": 5, "max_iterations": 20, "max_sub_agents": 20},  # Findings at limit
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_tasks(state)
        assert result == END
    
    def test_route_tasks_no_pending_tasks_routes_to_end(self):
        """Test that no pending tasks routes to END."""
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[],  # No tasks
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
        
        result = route_tasks(state)
        assert result == END
    
    def test_route_tasks_pending_tasks_creates_send_objects(self):
        """Test that pending tasks create Send objects for parallel execution."""
        task1 = ResearchTask(task_id="task1", topic="topic1", query="query1", priority=1)
        task2 = ResearchTask(task_id="task2", topic="topic2", query="query2", priority=1)
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[task1, task2],  # Two pending tasks
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
        
        result = route_tasks(state)
        
        # Should return list of Send objects
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(send, Send) for send in result)
    
    def test_route_tasks_filters_completed_tasks(self):
        """Test that completed tasks are filtered out."""
        task1 = ResearchTask(task_id="task1", topic="topic1", query="query1", priority=1)
        task2 = ResearchTask(task_id="task2", topic="topic2", query="query2", priority=1)
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[task1, task2],
            completed_tasks=["task1"],  # task1 completed
            failed_tasks=[],
            budget={"iterations": 1, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_tasks(state)
        
        # Should only create Send for task2
        assert isinstance(result, list)
        assert len(result) == 1
    
    def test_route_tasks_filters_failed_tasks(self):
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
            failed_tasks=["task1"],  # task1 failed
            budget={"iterations": 1, "max_iterations": 20, "max_sub_agents": 20},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = route_tasks(state)
        
        # Should only create Send for task2
        assert isinstance(result, list)
        assert len(result) == 1


class TestBuildResearchGraph:
    """Tests for build_research_graph function."""
    
    def test_build_research_graph_creates_compiled_graph(self):
        """Test that build_research_graph creates a compiled graph."""
        graph = build_research_graph()
        
        assert graph is not None
        # Graph should be compiled (has .invoke, .ainvoke methods)
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")
    
    def test_research_graph_has_required_nodes(self):
        """Test that graph has supervisor and sub_agent nodes."""
        graph = build_research_graph()
        
        # Check nodes exist in graph structure
        # LangGraph compiled graphs expose node names via .nodes
        node_names = list(graph.nodes.keys())
        assert "supervisor" in node_names
        assert "sub_agent" in node_names
