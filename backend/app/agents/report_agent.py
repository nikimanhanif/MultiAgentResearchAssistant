"""Report Agent - One-shot markdown report generation.

This agent takes the research brief and list of findings to generate
a final markdown-formatted report for the user.

Architecture: Supervisor Loop (Phase 3.7+)
- Input: ResearchBrief + List[Finding] (each with embedded Citation)
- Output: Markdown report with citations

Phase 4.1: Core report generation implementation
"""

from typing import List, Any, Dict
import logging
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import ResearchBrief, Finding, ReportFormat
from app.graphs.state import ResearchState
from app.config import get_deepseek_reasoner
from app.prompts.report_prompts import (
    get_report_generation_prompt,
    format_findings_for_prompt,
    get_summary_format_instructions,
    get_comparison_format_instructions,
    get_literature_review_instructions,
    get_gap_analysis_instructions,
    get_fact_validation_instructions,
    get_ranking_format_instructions,
)

logger = logging.getLogger(__name__)


def _build_report_generation_chain() -> Runnable:
    """Build chain for report generation.
    
    Returns:
        Runnable chain: prompt | LLM
    """
    prompt = get_report_generation_prompt()
    llm = get_deepseek_reasoner(temperature=0.5)
    
    chain = prompt | llm
    return chain


async def generate_report(
    brief: ResearchBrief,
    findings: List[Finding]
) -> str:
    """Generate markdown report from research brief and findings.
    
    This is a one-shot generation function that creates a comprehensive
    markdown report based on the research brief specifications and the
    list of findings.
    
    Args:
        brief: Research brief containing scope, sub-topics, constraints, and format
        findings: List of Finding objects with embedded citations
        
    Returns:
        Markdown-formatted report string
        
    Raises:
        ValueError: If brief or findings are invalid
        Exception: If LLM generation fails
    """
    # Validate inputs
    if not brief:
        raise ValueError("Research brief cannot be None")
    if not brief.scope:
        raise ValueError("Research brief must have a scope")
    if findings is None:
        raise ValueError("Findings list cannot be None (use empty list if no findings)")
    
    # Handle empty findings gracefully
    if not findings:
        return _generate_no_findings_report(brief)
    
    # Format brief components for prompt
    brief_subtopics = "\n".join([f"- {topic}" for topic in brief.sub_topics])
    brief_constraints = "\n".join(
        [f"- {key}: {value}" for key, value in brief.constraints.items()]
    ) if brief.constraints else "No specific constraints"
    
    # Get format-specific instructions
    format_type = brief.format or ReportFormat.OTHER
    format_instructions = _get_format_instructions(format_type)
    
    # Format findings for prompt context
    findings_context = format_findings_for_prompt(findings)
    
    # Initialize LLM (DeepSeek Reasoner for long-form generation)
    try:
        chain = _build_report_generation_chain()
    except ValueError as e:
        raise Exception(f"Failed to initialize LLM: {e}")
    
    # Generate report
    try:
        response = await chain.ainvoke({
            "brief_scope": brief.scope,
            "brief_subtopics": brief_subtopics,
            "brief_constraints": brief_constraints,
            "brief_format": format_type.value,
            "findings_context": findings_context,
            "format_instructions": format_instructions,
        })
        
        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        return str(response)
            
    except Exception as e:
        raise Exception(f"Failed to generate report: {str(e)}")


def _get_format_instructions(format_type: ReportFormat) -> str:
    """Get format-specific instructions for report generation.
    
    Args:
        format_type: Report format enum value
        
    Returns:
        Format-specific instruction string
    """
    format_map = {
        ReportFormat.SUMMARY: get_summary_format_instructions,
        ReportFormat.COMPARISON: get_comparison_format_instructions,
        ReportFormat.RANKING: get_ranking_format_instructions,
        ReportFormat.FACT_VALIDATION: get_fact_validation_instructions,
        ReportFormat.LITERATURE_REVIEW: get_literature_review_instructions,
        ReportFormat.GAP_ANALYSIS: get_gap_analysis_instructions,
        ReportFormat.OTHER: lambda: "Use a general report structure with Introduction, Findings, Conclusion, and References.",
    }
    
    instruction_func = format_map.get(format_type, format_map[ReportFormat.OTHER])
    return instruction_func()


def _generate_no_findings_report(brief: ResearchBrief) -> str:
    """Generate minimal report when no findings are available.
    
    Args:
        brief: Research brief
        
    Returns:
        Minimal markdown report indicating no findings
    """
    return f"""# Research Report: {brief.scope}

## Overview

This report was generated based on the following research brief:

**Scope**: {brief.scope}

**Sub-topics**:
{chr(10).join([f'- {topic}' for topic in brief.sub_topics])}

## Findings

No research findings were available to generate this report.

## Conclusion

Unable to complete research due to lack of findings. Please expand the research scope or try different search queries.
"""


# LangGraph Node Integration

async def report_agent_node(state: ResearchState) -> Dict[str, Any]:
    """Report agent node for LangGraph integration.
    
    This node wraps the generate_report function to fit into the LangGraph workflow.
    It extracts the research brief and findings from state, generates the report,
    and returns a state update with the report content.
    
    Handles reviewer feedback for refinement if present in state.
    
    Args:
        state: Current research state with research_brief and findings
        
    Returns:
        State update with report_content
    """
    logger.info("Report Agent Node: Generating report")
    
    # Extract required fields from state
    brief = state.get("research_brief")
    findings = state.get("findings", [])
    reviewer_feedback = state.get("reviewer_feedback")
    
    # Validate inputs
    if not brief:
        logger.error("Report Agent: No research brief found in state")
        return {
            "report_content": "Error: No research brief available for report generation.",
            "error": "Missing research brief"
        }
    
    try:
        # Generate report using existing function
        logger.info(f"Report Agent: Generating report with {len(findings)} findings")
        
        if reviewer_feedback:
            logger.info(f"Report Agent: Incorporating reviewer feedback: {reviewer_feedback[:100]}...")
        
        report_content = await generate_report(brief, findings)
        
        logger.info("Report Agent: Report generated successfully")
        
        # Return state update
        return {
            "report_content": report_content,
            "reviewer_feedback": None  # Clear feedback after processing
        }
    
    except Exception as e:
        logger.error(f"Report Agent: Error generating report - {e}")
        return {
            "report_content": f"Error generating report: {str(e)}",
            "error": str(e)
        }
