"""Supervisor Agent - LLM-based gap analysis and task generation.

This agent coordinates the research phase in the Supervisor Loop:
- Analyzes gaps in current findings vs research brief (LLM-driven)
- Generates new research tasks for missing topics
- Checks completion criteria or budget limits
- Filters and aggregates final findings

Architecture: Supervisor Loop
- Flat task queue
- Prompt-driven strategy selection
- LLM-based gap analysis 
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from pydantic import BaseModel, Field

from app.graphs.state import ResearchState
from app.models.schemas import ResearchTask, Finding
from app.config import get_deepseek_reasoner
from app.prompts.research_prompts import (
    SUPERVISOR_GAP_ANALYSIS_TEMPLATE,
    SUPERVISOR_FINDINGS_AGGREGATION_TEMPLATE,
)

logger = logging.getLogger(__name__)


class GapAnalysisOutput(BaseModel):
    """Structured output from supervisor gap analysis."""
    has_gaps: bool = Field(description="Whether significant gaps remain")
    is_complete: bool = Field(description="Whether research is complete")
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="List of identified gaps"
    )
    new_tasks: List[ResearchTask] = Field(
        default_factory=list,
        description="New tasks to address gaps"
    )
    reasoning: str = Field(description="Reasoning for the decision")


def supervisor_node(state: ResearchState) -> Dict[str, Any]:
    """Supervisor node for gap analysis and task generation.
    
    Responsibilities:
    1. Analyze current findings vs research brief (LLM-based)
    2. Generate new tasks for identified gaps
    3. Check completion criteria and budget limits
    4. Update iteration counter
    
    Args:
        state: Current research state with brief, findings, tasks, budget
        
    Returns:
        State update with new tasks or completion signal
    """
    logger.info("Supervisor: Starting gap analysis")
    
    # Extract state fields
    brief = state["research_brief"]
    findings = state.get("findings", [])
    completed_tasks = state.get("completed_tasks", [])
    failed_tasks = state.get("failed_tasks", [])
    budget = state["budget"]
    
    # Increment iteration counter
    current_iteration = budget["iterations"] + 1
    
    # Check hard budget limits first
    budget_exhausted = (
        current_iteration >= budget["max_iterations"] or
        len(findings) >= budget["max_sub_agents"]
    )
    
    if budget_exhausted:
        logger.info(f"Supervisor: Budget exhausted (iterations={current_iteration}, findings={len(findings)})")
        return {
            "budget": {**budget, "iterations": current_iteration},
            "is_complete": True
        }
    
    # Calculate current research statistics
    findings_by_topic = defaultdict(list)
    for finding in findings:
        findings_by_topic[finding.topic].append(finding)
    
    topics_covered = len(findings_by_topic)
    avg_credibility = (
        sum(f.credibility_score for f in findings) / len(findings)
        if findings else 0.0
    )
    
    # Prepare prompt inputs
    prompt_inputs = {
        "scope": brief.scope,
        "sub_topics": ", ".join(brief.sub_topics),
        "constraints": str(brief.constraints),
        "findings_count": len(findings),
        "topics_covered": topics_covered,
        "avg_credibility": avg_credibility,
        "iterations": current_iteration,
        "max_iterations": budget["max_iterations"],
        "total_sub_agents": len(findings),
        "max_sub_agents": budget["max_sub_agents"],
        "total_searches": budget.get("total_searches", 0),
        "completed_count": len(completed_tasks),
        "failed_tasks": ", ".join(failed_tasks) if failed_tasks else "None"
    }
    
    # Get LLM for structured output
    llm = get_deepseek_reasoner(temperature=0.5)
    structured_llm = llm.with_structured_output(GapAnalysisOutput)
    
    # Create chain and invoke
    chain = SUPERVISOR_GAP_ANALYSIS_TEMPLATE | structured_llm
    
    try:
        result: GapAnalysisOutput = chain.invoke(prompt_inputs)
        logger.info(f"Supervisor: Analysis complete - has_gaps={result.has_gaps}, is_complete={result.is_complete}")
        
        # Update state based on result
        state_update: Dict[str, Any] = {
            "budget": {**budget, "iterations": current_iteration},
            "gaps": {
                "has_gaps": result.has_gaps,
                "gaps_identified": result.gaps_identified,
                "reasoning": result.reasoning
            }
        }
        
        # Add new tasks if gaps exist and not complete
        if result.new_tasks and not result.is_complete:
            state_update["task_history"] = result.new_tasks
            logger.info(f"Supervisor: Generated {len(result.new_tasks)} new tasks")
        
        # Set completion flag
        state_update["is_complete"] = result.is_complete
        
        return state_update
        
    except Exception as e:
        logger.error(f"Supervisor gap analysis failed: {e}")
        # On error, mark as complete to avoid infinite loops
        return {
            "budget": {**budget, "iterations": current_iteration},
            "is_complete": True,
            "error": f"Supervisor analysis failed: {str(e)}"
        }


def aggregate_findings(state: ResearchState) -> List[Finding]:
    """Aggregate and filter findings for report generation (Phase 8.5).
    
    Applies LLM-based filtering and programmatic deduplication.
    
    Args:
        state: Research state with findings and brief
        
    Returns:
        Filtered and ranked list of findings
    """
    findings = state.get("findings", [])
    brief = state["research_brief"]
    
    if not findings:
        return []
    
    # Step 1: Programmatic filtering (credibility threshold)
    filtered = [f for f in findings if f.credibility_score >= 0.5]
    logger.info(f"Filtered {len(findings)} -> {len(filtered)} findings (credibility >= 0.5)")
    
    # Step 2: Programmatic deduplication
    seen_dois = {}
    seen_urls = {}
    seen_titles = {}
    deduplicated = []
    
    for finding in filtered:
        citation = finding.citation
        
        # Check DOI first (most reliable)
        if citation.doi:
            if citation.doi in seen_dois:
                # Keep higher credibility
                if finding.credibility_score > seen_dois[citation.doi].credibility_score:
                    deduplicated.remove(seen_dois[citation.doi])
                    deduplicated.append(finding)
                    seen_dois[citation.doi] = finding
                continue
            seen_dois[citation.doi] = finding
            deduplicated.append(finding)
            continue
        
        # Check URL
        if citation.url:
            if citation.url in seen_urls:
                if finding.credibility_score > seen_urls[citation.url].credibility_score:
                    deduplicated.remove(seen_urls[citation.url])
                    deduplicated.append(finding)
                    seen_urls[citation.url] = finding
                continue
            seen_urls[citation.url] = finding
            deduplicated.append(finding)
            continue
        
        # Check title + first author
        if citation.title and citation.authors:
            key = (citation.title, citation.authors[0] if citation.authors else "")
            if key in seen_titles:
                if finding.credibility_score > seen_titles[key].credibility_score:
                    deduplicated.remove(seen_titles[key])
                    deduplicated.append(finding)
                    seen_titles[key] = finding
                continue
            seen_titles[key] = finding
            deduplicated.append(finding)
            continue
        
        # No deduplication possible, add it
        deduplicated.append(finding)
    
    logger.info(f"Deduplicated {len(filtered)} -> {len(deduplicated)} findings")
    
    # Step 3: Sort by credibility (highest first)
    sorted_findings = sorted(
        deduplicated,
        key=lambda f: f.credibility_score,
        reverse=True
    )
    
    return sorted_findings
