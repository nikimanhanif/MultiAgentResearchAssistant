"""Unit tests for supervisor agent findings aggregation."""

import pytest
from typing import List

from app.agents.supervisor_agent import aggregate_findings
from app.graphs.state import ResearchState
from app.models.schemas import ResearchBrief, Finding, Citation, SourceType


class TestAggregateFindingsFiltering:
    """Tests for credibility-based filtering."""
    
    def test_filters_low_credibility_findings(self):
        """Test that findings with credibility < 0.5 are filtered out."""
        findings = [
            Finding(
                claim="high credibility",
                citation=Citation(source="source1", url="http://example.com/1", title="title1"),
                topic="test",
                credibility_score=0.8
            ),
            Finding(
                claim="low credibility",
                citation=Citation(source="source2", url="http://example.com/2", title="title2"),
                topic="test",
                credibility_score=0.3  # Below threshold
            ),
            Finding(
                claim="medium credibility",
                citation=Citation(source="source3", url="http://example.com/3", title="title3"),
                topic="test",
                credibility_score=0.6
            ),
        ]
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=findings,
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = aggregate_findings(state)
        
        # Should only include findings with credibility >= 0.5
        assert len(result) == 2
        assert all(f.credibility_score >= 0.5 for f in result)
    
    def test_empty_findings_returns_empty_list(self):
        """Test that empty findings list returns empty result."""
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=[],
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = aggregate_findings(state)
        assert result == []


class TestAggregateFindingsDeduplication:
    """Tests for programmatic deduplication logic."""
    
    def test_deduplicates_by_doi(self):
        """Test that duplicate DOIs are removed."""
        findings = [
            Finding(
                claim="first",
                citation=Citation(
                    source="source1",
                    url="http://example.com/1",
                    title="paper1",
                    doi="10.1234/test",
                    credibility_score=0.8
                ),
                topic="test",
                credibility_score=0.8
            ),
            Finding(
                claim="second",
                citation=Citation(
                    source="source2",
                    url="http://example.com/2",
                    title="paper2",
                    doi="10.1234/test",  # Same DOI
                    credibility_score=0.7
                ),
                topic="test",
                credibility_score=0.7
            ),
        ]
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=findings,
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = aggregate_findings(state)
        
        # Should deduplicate to 1 finding (higher credibility)
        assert len(result) == 1
        assert result[0].credibility_score == 0.8  # Keeps higher credibility
    
    def test_deduplicates_by_url(self):
        """Test that duplicate URLs are removed."""
        findings = [
            Finding(
                claim="first",
                citation=Citation(
                    source="source1",
                    url="http://example.com/same",
                    title="title1",
                    credibility_score=0.6
                ),
                topic="test",
                credibility_score=0.6
            ),
            Finding(
                claim="second",
                citation=Citation(
                    source="source2",
                    url="http://example.com/same",  # Same URL
                    title="title2",
                    credibility_score=0.9
                ),
                topic="test",
                credibility_score=0.9
            ),
        ]
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=findings,
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = aggregate_findings(state)
        
        # Should deduplicate to 1 finding (higher credibility)
        assert len(result) == 1
        assert result[0].credibility_score == 0.9
    
    def test_deduplicates_by_title_and_author(self):
        """Test that duplicate title+author combinations are removed."""
        findings = [
            Finding(
                claim="first",
                citation=Citation(
                    source="source1",
                    url="http://example.com/1",
                    title="Same Title",
                    authors=["Author A", "Author B"],
                    credibility_score=0.7
                ),
                topic="test",
                credibility_score=0.7
            ),
            Finding(
                claim="second",
                citation=Citation(
                    source="source2",
                    url="http://example.com/2",
                    title="Same Title",
                    authors=["Author A", "Author C"],  # Same first author + title
                    credibility_score=0.8
                ),
                topic="test",
                credibility_score=0.8
            ),
        ]
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=findings,
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = aggregate_findings(state)
        
        # Should deduplicate by title+first_author
        assert len(result) == 1
        assert result[0].credibility_score == 0.8  # Keeps higher credibility
    
    def test_keeps_unique_findings(self):
        """Test that unique findings are all preserved."""
        findings = [
            Finding(
                claim="first",
                citation=Citation(
                    source="source1",
                    url="http://example.com/1",
                    title="title1",
                    authors=["Author A"],
                    doi="10.1234/a",
                    credibility_score=0.8
                ),
                topic="test",
                credibility_score=0.8
            ),
            Finding(
                claim="second",
                citation=Citation(
                    source="source2",
                    url="http://example.com/2",
                    title="title2",
                    authors=["Author B"],
                    doi="10.1234/b",
                    credibility_score=0.7
                ),
                topic="test",
                credibility_score=0.7
            ),
            Finding(
                claim="third",
                citation=Citation(
                    source="source3",
                    url="http://example.com/3",
                    title="title3",
                    authors=["Author C"],
                    doi="10.1234/c",
                    credibility_score=0.9
                ),
                topic="test",
                credibility_score=0.9
            ),
        ]
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=findings,
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = aggregate_findings(state)
        
        # All unique findings should be preserved
        assert len(result) == 3


class TestAggregateFindingsSorting:
    """Tests for credibility-based sorting."""
    
    def test_sorts_by_credibility_descending(self):
        """Test that findings are sorted by credibility (highest first)."""
        findings = [
            Finding(
                claim="low",
                citation=Citation(source="source1", url="http://example.com/1", title="title1"),
                topic="test",
                credibility_score=0.5
            ),
            Finding(
                claim="high",
                citation=Citation(source="source2", url="http://example.com/2", title="title2"),
                topic="test",
                credibility_score=0.9
            ),
            Finding(
                claim="medium",
                citation=Citation(source="source3", url="http://example.com/3", title="title3"),
                topic="test",
                credibility_score=0.7
            ),
        ]
        
        state = ResearchState(
            research_brief=ResearchBrief(
                scope="test", sub_topics=[], constraints={}, format=None, deliverables="test report"
            ),
            findings=findings,
            task_history=[],
            completed_tasks=[],
            failed_tasks=[],
            budget={},
            is_complete=False,
            gaps=None,
            error=None,
            messages=[],
            report_content="",
            reviewer_feedback=None
        )
        
        result = aggregate_findings(state)
        
        # Should be sorted descending by credibility
        assert len(result) == 3
        assert result[0].credibility_score == 0.9
        assert result[1].credibility_score == 0.7
        assert result[2].credibility_score == 0.5
