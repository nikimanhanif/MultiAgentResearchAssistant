"""Prompt templates for Report Agent.

This module will contain all prompt templates used by the Report Agent for
generating various types of research reports (summary, comparison, ranking,
literature review, gap analysis, etc.).

All prompts use LangChain's ChatPromptTemplate for consistent message formatting
and follow best practices from the LangChain documentation.

Note: Full implementation pending Phase 4. This file provides the structure
and placeholder prompts for future implementation.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Base Report Generation Prompt (Phase 4.1)
# Required inputs: research_brief (str), summarized_findings (str), format_type (str)
REPORT_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research report generator. Your job is to create a comprehensive, well-structured markdown report from research findings.

General Guidelines:
- Write in clear, professional academic language
- Structure content logically with proper headings
- Include all relevant citations
- Format in markdown with proper syntax
- Follow the specified report format

Report Components:
1. Introduction/Context
2. Main findings organized by sub-topic
3. Analysis and synthesis
4. Conclusions
5. References/Bibliography

Citation Requirements:
- Include in-text citations for all claims
- Format: [1], [2], etc.
- Full bibliography at the end
- Include credibility indicators where relevant

Generate a complete report following the specified format."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Research Brief:
{research_brief}

Summarized Findings:
{summarized_findings}

Report Format: {format_type}

Generate a comprehensive markdown report."""
    ),
])


# Summary Report Format Instructions (Phase 4.2)
def get_summary_format_instructions() -> str:
    """Get instructions for summary report format.
    
    Returns:
        Formatted instruction string for summary reports
    """
    return """
Summary Report Format:
- Start with executive summary (2-3 paragraphs)
- Organize findings by sub-topic
- Highlight key takeaways (bullet points)
- Include brief conclusion
- Add complete references section

Structure:
# [Topic] - Research Summary

## Executive Summary
[2-3 paragraph overview]

## Key Findings
### [Sub-topic 1]
[Findings with citations]

### [Sub-topic 2]
[Findings with citations]

## Conclusion
[Brief synthesis]

## References
[1] Citation 1
[2] Citation 2
"""


# Comparison Report Format Instructions (Phase 4.2)
def get_comparison_format_instructions() -> str:
    """Get instructions for comparison report format.
    
    Returns:
        Formatted instruction string for comparison reports
    """
    return """
Comparison Report Format:
- Introduction explaining comparison criteria
- Side-by-side comparison table
- Detailed analysis of each item
- Recommendations based on comparison
- Complete references

Structure:
# [Topic] - Comparative Analysis

## Introduction
[Context and comparison criteria]

## Comparison Overview
| Criterion | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| ...       | ...      | ...      | ...      |

## Detailed Analysis
### Option A
[Analysis with citations]

### Option B
[Analysis with citations]

## Recommendations
[Evidence-based recommendations]

## References
[Complete bibliography]
"""


# Literature Review Format Instructions (Phase 4.2)
def get_literature_review_instructions() -> str:
    """Get instructions for literature review format.
    
    Returns:
        Formatted instruction string for literature review reports
    """
    return """
Literature Review Format:
- Introduction with research context
- Thematic sections organized by sub-topics
- Critical analysis of sources
- Identification of research gaps
- Conclusion with synthesis
- Bibliography with credibility indicators

Structure:
# [Topic] - Literature Review

## Introduction
[Research context and scope]

## Literature Analysis
### [Theme/Sub-topic 1]
[Review of relevant literature with citations and analysis]

### [Theme/Sub-topic 2]
[Review of relevant literature with citations and analysis]

## Research Gaps
[Identified gaps in current research]

## Conclusion
[Synthesis and future directions]

## Bibliography
[1] Citation with credibility indicator (Score: 0.85)
[2] Citation with credibility indicator (Score: 0.92)
"""


# Gap Analysis Format Instructions (Phase 4.2)
def get_gap_analysis_instructions() -> str:
    """Get instructions for gap analysis format.
    
    Returns:
        Formatted instruction string for gap analysis reports
    """
    return """
Gap Analysis Format:
- Executive summary of gaps found
- Coverage analysis (what's well-covered vs under-researched)
- Gap categorization (coverage, depth, temporal, perspective)
- Recommendations for future research
- Complete references

Structure:
# [Topic] - Research Gap Analysis

## Executive Summary
[Overview of gap analysis findings]

## Coverage Analysis
### Well-Covered Areas
[Topics with substantial research]

### Under-Researched Areas
[Topics needing more investigation]

## Gap Identification
### Coverage Gaps
[Missing topics or perspectives]

### Depth Gaps
[Areas needing deeper investigation]

### Temporal Gaps
[Missing time periods or recent developments]

## Recommendations
[Specific recommendations for future research]

## References
[Complete bibliography with credibility indicators]
"""


# Fact Validation Format Instructions (Phase 4.2)
def get_fact_validation_instructions() -> str:
    """Get instructions for fact validation format.
    
    Returns:
        Formatted instruction string for fact validation reports
    """
    return """
Fact Validation Format:
- Introduction explaining validation criteria
- Each claim with validation results
- Credibility assessment for each source
- Overall conclusion
- Complete references with credibility scores

Structure:
# [Topic] - Fact Validation Report

## Introduction
[Validation criteria and methodology]

## Claim Validation
### Claim 1: [Statement]
**Validation Result**: ✅ Supported / ⚠️ Partially Supported / ❌ Not Supported
**Evidence**: [Citations with credibility scores]
**Analysis**: [Explanation]

### Claim 2: [Statement]
[Same structure]

## Overall Assessment
[Summary of validation results]

## References
[1] Source 1 (Credibility: 0.92 - High)
[2] Source 2 (Credibility: 0.65 - Medium)
"""


# Ranking Report Format Instructions (Phase 4.2)
def get_ranking_format_instructions() -> str:
    """Get instructions for ranking report format.
    
    Returns:
        Formatted instruction string for ranking reports
    """
    return """
Ranking Report Format:
- Introduction with ranking criteria
- Ranked list with justification
- Detailed analysis for each item
- Summary table
- Complete references

Structure:
# [Topic] - Ranking Analysis

## Introduction
[Ranking criteria and methodology]

## Rankings

### 1. [Top Item]
**Score**: X/10
**Strengths**: [With citations]
**Weaknesses**: [With citations]
**Justification**: [Why ranked #1]

### 2. [Second Item]
[Same structure]

## Summary Table
| Rank | Item | Score | Key Strength |
|------|------|-------|--------------|
| 1    | ...  | X/10  | ...          |

## References
[Complete bibliography]
"""


# NOTE: Additional prompts for Phase 4 will be added during implementation:
# - Format-specific prompt variants
# - Citation formatting prompts
# - Structure validation prompts

