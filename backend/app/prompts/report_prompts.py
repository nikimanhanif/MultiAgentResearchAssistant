"""
Prompt templates for Report Agent.

Contains prompt templates for generating various types of research reports
(summary, comparison, ranking, literature review, gap analysis, etc.)
using LangChain's ChatPromptTemplate.
"""

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import Finding


def format_findings_for_prompt(findings: List[Finding]) -> str:
    """
    Format findings list into numbered context for prompt.
    
    Args:
        findings: List of Finding objects with embedded citations.
        
    Returns:
        str: Formatted string with numbered findings and citation details.
    """
    if not findings:
        return "No findings available."
    
    formatted_lines = []
    for idx, finding in enumerate(findings):
        citation = finding.citation
        
        authors_str = "Unknown Author"
        if citation.authors:
            authors_str = ", ".join(citation.authors)
        
        credibility_warning = ""
        if finding.credibility_score < 0.5:
            credibility_warning = f" ⚠️ LOW CREDIBILITY ({finding.credibility_score:.2f})"
        
        finding_entry = f"""[{idx}] {finding.claim}
   Topic: {finding.topic}
   Source: {citation.title or citation.source}
   Authors: {authors_str}
   URL: {citation.url or 'N/A'}
   Credibility Score: {finding.credibility_score:.2f}{credibility_warning}
   Source Type: {citation.source_type or 'unknown'}"""
        
        if citation.year:
            finding_entry += f"\n   Year: {citation.year}"
        if citation.doi:
            finding_entry += f"\n   DOI: {citation.doi}"
        
        formatted_lines.append(finding_entry)
    
    return "\n\n".join(formatted_lines)


def get_report_generation_prompt() -> ChatPromptTemplate:
    """
    Create report generation prompt template.
    
    Returns:
        ChatPromptTemplate: Configured for report generation.
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert research report writer specializing in academic and professional reports.

Your task is to generate a comprehensive, well-structured markdown report from the provided research findings.

## CRITICAL CITATION RULES:

1. **In-text Citations**: 
   - Use ONLY the finding index numbers [0], [1], [2], etc. provided in the findings list
   - Every claim MUST reference its source using the correct finding index
   - Multiple claims from the same source should repeat the citation
   - Example: "Machine learning models can be fine-tuned [0]. This approach is common [0]."

2. **Bibliography/References Section**:
   - Include ALL findings at the end in a "References" or "Bibliography" section
   - Format each reference with full citation details:
     * Authors (if available)
     * Title
     * Source/Publication
     * Year (if available)
     * DOI (if available)
     * URL
     * **Credibility Indicator**: (Score: X.XX)
   - Add ⚠️ WARNING for sources with credibility score < 0.5
   - Example format:
     ```
     [0] Smith, J., Doe, A. (2023). "Title of Paper". Nature. DOI: 10.1038/example
         https://nature.com/article (Credibility: 0.95 - High)
     
     [1] Unknown Author. "Blog Post Title". Personal Blog. https://example.com
         ⚠️ (Credibility: 0.35 - Low Quality Source)
     ```

3. **Credibility Warnings**:
   - For any source with credibility < 0.5, include a warning in the bibliography
   - Consider mentioning low credibility in-text when critical claims rely on weak sources

## REPORT STRUCTURE:

1. **Title**: Clear, descriptive title based on research scope
2. **Introduction**: Context and objectives (2-3 paragraphs)
3. **Main Content**: Organized by sub-topics with proper headings
4. **Findings/Analysis**: Present research findings with citations
5. **Conclusion**: Synthesis and key takeaways
6. **References/Bibliography**: Complete citation list with credibility indicators

## MARKDOWN FORMATTING:

- Use proper heading hierarchy (# ## ###)
- Use **bold** for emphasis, *italic* for technical terms
- Use bullet points and numbered lists for clarity
- Use tables for comparisons (if format requires)
- Use code blocks for technical content (if relevant)
- Ensure all markdown is valid and renders correctly

## QUALITY STANDARDS:

- Write in clear, professional academic language
- Maintain logical flow and coherence
- Synthesize information, don't just list findings
- Provide analysis and insight where appropriate
- Follow the specified report format exactly
- Ensure all claims are supported by citations

DO NOT hallucinate or invent information beyond what's in the findings list.
DO NOT create fake citations or references.
ONLY use the finding indices [0], [1], [2]... provided in the findings context."""),
        
        ("human", """# Research Brief

**Scope**: {brief_scope}

**Sub-topics to Cover**:
{brief_subtopics}

**Constraints**:
{brief_constraints}

**Report Format**: {brief_format}

---

# Reviewer Feedback (Optional)

{reviewer_feedback}

---

# Findings Context

{findings_context}

---

# Format-Specific Instructions

{format_instructions}

---

Please generate a complete markdown report following the above requirements. If reviewer_feedback is provided, adjust the report to address the specific feedback while maintaining the original requirements.""")
    ])


def get_summary_format_instructions() -> str:
    """
    Get instructions for summary report format.
    
    Returns:
        str: Formatted instruction string for summary reports.
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


def get_comparison_format_instructions() -> str:
    """
    Get instructions for comparison report format.
    
    Returns:
        str: Formatted instruction string for comparison reports.
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


def get_literature_review_instructions() -> str:
    """
    Get instructions for literature review format.
    
    Returns:
        str: Formatted instruction string for literature review reports.
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


def get_gap_analysis_instructions() -> str:
    """
    Get instructions for gap analysis format.
    
    Returns:
        str: Formatted instruction string for gap analysis reports.
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


def get_fact_validation_instructions() -> str:
    """
    Get instructions for fact validation format.
    
    Returns:
        str: Formatted instruction string for fact validation reports.
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


def get_ranking_format_instructions() -> str:
    """
    Get instructions for ranking report format.
    
    Returns:
        str: Formatted instruction string for ranking reports.
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

