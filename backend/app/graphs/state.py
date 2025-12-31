"""
LangGraph state definition for research workflow.

Defines the state structure for the research agent's LangGraph workflow,
utilizing the Supervisor Loop pattern with task queue and parallel execution.

Key Features:
- Reducer Pattern: findings, task_history, completed_tasks, and messages use operator.add.
- Task Queue: Supervisor generates tasks, sub-agents execute in parallel.
- Budget Limits: Enforces iteration and search limits.
- LLM-Based Gap Analysis: Supervisor uses LLM prompts for decision making.
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator
from app.models.schemas import (
    ResearchBrief,
    Finding,
    ResearchTask,
    SubAgentSummary,
)


def merge_budgets(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """
    Custom reducer for budget field to handle concurrent updates.
    
    Handles parallel sub-agent search count updates while preserving
    supervisor iteration counter precedence.
    
    Args:
        left: Current budget state from the graph.
        right: Budget update from a node.
        
    Returns:
        Dict[str, int]: Merged budget dictionary.
    """
    merged = right.copy()
    
    merged["iterations"] = max(
        left.get("iterations", 0), 
        right.get("iterations", 0)
    )
    
    left_searches = left.get("total_searches", 0)
    right_searches = right.get("total_searches", 0)
    searches_delta = max(0, right_searches - left_searches)
    merged["total_searches"] = left_searches + searches_delta
    
    return merged


def merge_findings_with_dedup(left: List[Finding], right: List[Finding]) -> List[Finding]:
    """
    Custom reducer for findings that deduplicates on merge.
    
    Deduplication is based on DOI, URL, or title+author matching.
    When duplicates are found, keeps the finding with higher credibility.
    """
    if not left:
        return right
    if not right:
        return left
    
    # Build lookup indices from existing findings
    seen_dois: Dict[str, Finding] = {}
    seen_urls: Dict[str, Finding] = {}
    seen_titles: Dict[tuple, Finding] = {}
    result = []
    
    for finding in left:
        citation = finding.citation
        if citation.doi:
            seen_dois[citation.doi] = finding
        if citation.url:
            seen_urls[citation.url] = finding
        if citation.title and citation.authors:
            key = (citation.title, citation.authors[0] if citation.authors else "")
            seen_titles[key] = finding
        result.append(finding)
    
    # Add new findings, checking for duplicates
    for finding in right:
        citation = finding.citation
        existing = None
        
        # Check DOI
        if citation.doi and citation.doi in seen_dois:
            existing = seen_dois[citation.doi]
        # Check URL
        elif citation.url and citation.url in seen_urls:
            existing = seen_urls[citation.url]
        # Check title+author
        elif citation.title and citation.authors:
            key = (citation.title, citation.authors[0] if citation.authors else "")
            if key in seen_titles:
                existing = seen_titles[key]
        
        if existing:
            # Keep higher credibility finding
            if finding.credibility_score > existing.credibility_score:
                result.remove(existing)
                result.append(finding)
                # Update indices
                if citation.doi:
                    seen_dois[citation.doi] = finding
                if citation.url:
                    seen_urls[citation.url] = finding
                if citation.title and citation.authors:
                    key = (citation.title, citation.authors[0] if citation.authors else "")
                    seen_titles[key] = finding
        else:
            result.append(finding)
            # Add to indices
            if citation.doi:
                seen_dois[citation.doi] = finding
            if citation.url:
                seen_urls[citation.url] = finding
            if citation.title and citation.authors:
                key = (citation.title, citation.authors[0] if citation.authors else "")
                seen_titles[key] = finding
    
    return result


class ResearchState(TypedDict, total=False):
    """
    State schema for research workflow graph (Supervisor Loop).
    
    Fields using reducers support parallel updates from multiple nodes.
    The findings field uses a custom reducer that deduplicates on merge.
    """
    research_brief: ResearchBrief
    findings: Annotated[List[Finding], merge_findings_with_dedup]
    task_history: Annotated[List[ResearchTask], operator.add]
    completed_tasks: Annotated[List[str], operator.add]
    failed_tasks: Annotated[List[str], operator.add]
    sub_agent_summaries: Annotated[List[SubAgentSummary], operator.add]
    budget: Annotated[Dict[str, int], merge_budgets]
    gaps: Optional[Dict[str, Any]]
    is_complete: bool
    error: Annotated[List[str], operator.add]
    messages: Annotated[List[Dict[str, Any]], operator.add]
    report_content: str
    reviewer_feedback: Optional[str]
    scope_clarification_rounds: int 


class SubAgentState(ResearchState):
    """State schema for individual sub-agent execution with task assignment."""
    task: ResearchTask


def create_initial_state(research_brief: ResearchBrief) -> ResearchState:
    """
    Create initial research state from research brief.
    
    Args:
        research_brief: Research brief from scope agent.
        
    Returns:
        ResearchState: Initial state with default values.
    """
    return ResearchState(
        research_brief=research_brief,
        findings=[],
        task_history=[],
        completed_tasks=[],
        failed_tasks=[],
        sub_agent_summaries=[],
        budget={
            "iterations": 0,
            "max_iterations": 20,
            "max_sub_agents": 20,
            "max_searches_per_agent": 2,
            "total_searches": 0
        },
        gaps=None,
        is_complete=False,
        error=[],
        messages=[],
        report_content="",
        reviewer_feedback=None
    )
