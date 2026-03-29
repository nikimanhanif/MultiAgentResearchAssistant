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
    for idx, finding in enumerate(findings, 1):
        citation = finding.citation
        
        if citation.authors:
            authors_str = ", ".join(citation.authors)
        elif citation.venue:
            authors_str = citation.venue
        elif citation.source:
            authors_str = citation.source
        else:
            authors_str = "Unattributed Source"
        
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
   - Use ONLY the finding index numbers [1], [2], [3], etc. provided in the findings list
   - Every claim MUST reference its source using the correct finding index
   - Multiple claims from the same source should repeat the citation
   - Example: "Machine learning models can be fine-tuned [1]. This approach is common [1]."

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
     [1] Smith, J., Doe, A. (2023). "Title of Paper". Nature. DOI: 10.1038/example
         https://nature.com/article (Credibility: 0.95 - High)
     
     [2] MIT Technology Review. "Blog Post Title". MIT Technology Review.
         https://technologyreview.com/article (Credibility: 0.70 - Established Tech Blog)
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

## IN-DEPTH ANALYSIS REQUIREMENTS:

This is an academic-quality report. Avoid surface-level summaries. Each section should demonstrate:

1. **Comprehensive Coverage**: Use ALL relevant findings. The research effort to gather these was significant.
2. **Critical Synthesis**: Don't just list what papers found — compare methodologies, note where studies agree/disagree
3. **Depth Per Section**: Each thematic section should be 3-4 paragraphs minimum:
   - Opening paragraph: Frame the theme and its importance
   - Body paragraphs: Discuss 2-3 key papers with methodology, results, and limitations
   - Synthesis paragraph: Comparative analysis and emergent patterns
4. **Contextual Analysis**: Explain WHY findings matter, not just WHAT they say
5. **Acknowledge Limitations**: Note study limitations and areas of uncertainty

Handling findings:
- Include findings even if tangentially related — they add context
- When multiple findings cover the same point, synthesize them rather than picking one
- Discuss methodology and credibility differences between sources
- Include ALL relevant findings in the References section

DO NOT hallucinate or invent information beyond what's in the findings list.
DO NOT create fake citations or references.
ONLY use the finding indices [1], [2], [3]... provided in the findings context."""),
        
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


def get_literature_review_instructions() -> str:
    """
    Get instructions for literature review format.
    
    Returns:
        str: Formatted instruction string for academic literature review reports.
    """
    return """
Literature Review Format - ACADEMIC DEPTH REQUIRED

This is an academic literature review. Each section must demonstrate scholarly rigor.

Structure:
# [Topic] - Literature Review

## Introduction (2-3 paragraphs)
- Research context and significance
- Define the scope and objectives of the review
- Outline the structure of the review

## Literature Analysis
For EACH thematic section, provide 3-4 paragraphs:

### [Theme/Sub-topic 1]
**Paragraph 1 - Framing**: What is this theme about? Why is it important to the research question?

**Paragraph 2-3 - Key Papers**: For 2-3 key papers, discuss:
- What methodology did they use?
- What were their key findings?
- What are the limitations of their approach?

**Paragraph 4 - Synthesis**: 
- Where do these studies agree or disagree?
- What patterns emerge across the literature?
- What gaps remain within this theme?

### [Theme/Sub-topic 2]
[Same 4-paragraph structure]

## Research Gaps (1-2 paragraphs per gap)
- Identify specific gaps in current research
- Explain why each gap matters
- Suggest potential research directions

## Conclusion (2-3 paragraphs)
- Synthesize key findings across all themes
- Discuss implications for the field
- Recommend future research directions

## Bibliography
[1] Full citation with authors, year, title, publication (Credibility: X.XX)
[2] ...
"""


def get_deep_research_instructions() -> str:
    """
    Get instructions for deep research format.
    
    Returns:
        str: Formatted instruction string for in-depth research reports.
    """
    return """
Deep Research Format - COMPREHENSIVE ANALYSIS REQUIRED

This format is for thorough investigation of a specific topic or question.

Structure:
# [Topic] - Research Report

## Executive Summary (2-3 paragraphs)
- Key findings and conclusions upfront
- Significance of the findings
- Brief methodology overview

## Background & Context (2-3 paragraphs)
- Why this topic matters
- Current state of knowledge
- Key definitions and framework

## Methodology
- How the research was conducted
- What sources were consulted
- Limitations of the approach

## Findings
For EACH major finding area, provide in-depth analysis:

### [Finding Area 1]
**Context**: Why this finding area is important

**Evidence**: Present 2-3 key sources with:
- What they found (methodology and results)
- Credibility assessment
- How they relate to each other

**Analysis**: 
- Synthesize across sources
- Discuss implications
- Note any contradictions or uncertainties

### [Finding Area 2]
[Same structure]

## Discussion (2-3 paragraphs)
- What do the findings mean collectively?
- How do they answer the research question?
- What remains uncertain?

## Conclusion
- Summarize key takeaways
- Practical implications
- Recommendations

## References
[Full bibliography with credibility indicators]
"""


def get_comparative_instructions() -> str:
    """
    Get instructions for comparative analysis format.
    
    Returns:
        str: Formatted instruction string for comparison reports.
    """
    return """
Comparative Analysis Format - BALANCED EVALUATION REQUIRED

This format compares multiple options, approaches, or technologies.

Structure:
# [Topic] - Comparative Analysis

## Introduction (2 paragraphs)
- Context for the comparison
- Explain the comparison criteria and why they matter

## Comparison Framework
| Criterion | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| ...       | ...      | ...      | ...      |

## Detailed Analysis
For EACH option, provide 3-4 paragraphs:

### Option A: [Name]
**Overview**: What is this option? Who uses it?

**Strengths** (with citations):
- Detailed discussion of advantages
- Evidence from research/case studies

**Weaknesses** (with citations):
- Honest assessment of limitations
- Specific scenarios where it fails

**Best Use Cases**:
- When to choose this option
- Who benefits most

### Option B: [Name]
[Same structure]

## Cross-Cutting Analysis (2-3 paragraphs)
- How do the options compare on the most important criteria?
- What trade-offs exist between options?
- Are there scenarios where each option is clearly better?

## Recommendations
- Specific recommendations based on use case
- Decision framework for readers
- Key factors to consider

## References
[Full bibliography]
"""


def get_gap_analysis_instructions() -> str:
    """
    Get instructions for gap analysis format.
    
    Returns:
        str: Formatted instruction string for research gap analysis reports.
    """
    return """
Gap Analysis Format - SYSTEMATIC IDENTIFICATION REQUIRED

This format identifies what's missing or under-researched in a field.

Structure:
# [Topic] - Research Gap Analysis

## Executive Summary (2 paragraphs)
- Overview of the analysis scope
- Key gaps identified (preview)

## Current State of Research

### Well-Covered Areas (2-3 paragraphs per area)
For each well-covered area:
- What topics have substantial research?
- What methodologies are commonly used?
- What consensus exists?
- Cite key papers demonstrating coverage

### Moderately-Covered Areas (1-2 paragraphs per area)
- Topics with some research but room for more
- What's been done vs. what's missing

## Gap Identification

### Coverage Gaps (2-3 paragraphs)
- What topics are missing entirely?
- What perspectives are underrepresented?
- Why might these gaps exist?

### Methodological Gaps (1-2 paragraphs)
- What approaches haven't been tried?
- What data sources are underutilized?

### Temporal Gaps (1-2 paragraphs)
- What time periods are understudied?
- What recent developments need investigation?

### Applied/Practical Gaps (1-2 paragraphs)
- Where is the theory-practice disconnect?
- What implementation studies are needed?

## Prioritized Research Agenda
- Rank gaps by importance/feasibility
- Suggest specific research questions
- Identify potential methodologies

## Conclusion
- Summary of most critical gaps
- Call to action for researchers

## References
[Full bibliography with credibility indicators]
"""


