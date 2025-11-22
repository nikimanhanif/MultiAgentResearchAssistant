"""Prompt templates for Research Agent.

This module will contain all prompt templates used by the Research Agent,
including supervisor coordination, sub-agent task delegation, and findings
aggregation.

All prompts use LangChain's ChatPromptTemplate for consistent message formatting
and follow best practices from the LangChain documentation.

Note: Full implementation pending Phase 8. This file provides the structure
and placeholder prompts for future implementation.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Strategy Selection Prompt (Phase 8.1)
# Required inputs: research_brief (str), sub_topic_count (int)
RESEARCH_STRATEGY_SELECTION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research strategy planner. Your job is to analyze a research brief and determine the optimal research execution strategy.

Select one of three modes:
1. FLAT - Simple queries with 3-5 sub-agents (no hierarchy)
2. FLAT_REFINEMENT - Medium complexity with 3-5 sub-agents + gap-based refinement
3. HIERARCHICAL - Complex queries with 2-level tree structure (5-8 L1 + 10-15 L2 sub-agents)

Consider:
- Number of sub-topics (more topics → hierarchical)
- Research depth requirements (deeper → hierarchical)
- Complexity of the research question
- Expected token budget

Return response as JSON:
{{
    "mode": "FLAT/FLAT_REFINEMENT/HIERARCHICAL",
    "reasoning": "Explanation for your choice",
    "expected_sub_agents": <number>,
    "depth_required": <1 or 2>
}}

Return ONLY the JSON object, no other text."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Research Brief:
{research_brief}

Number of Sub-topics: {sub_topic_count}

Determine the optimal research strategy."""
    ),
])


# Task Decomposition Prompt (Phase 8.2)
# Required inputs: research_brief (str), strategy (str)
RESEARCH_TASK_DECOMPOSITION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research task decomposition specialist. Your job is to break down a research brief into distinct, non-overlapping sub-agent tasks.

Create tasks that are:
- Distinct and non-overlapping (no redundant work)
- Clearly scoped (each task focuses on one sub-topic or aspect)
- Actionable (sub-agents can execute them with search tools)

For each task, provide:
- Clear description of what to research
- Query variants (2-3 alternative phrasings)
- Preferred tools (academic vs web search)
- Context budget (2K-6K tokens)

Return response as JSON array:
[
    {{
        "task_id": "task_1",
        "description": "Research sub-topic X focusing on aspect Y",
        "query_variants": ["query 1", "query 2", "query 3"],
        "preferred_tools": ["tavily_search", "arxiv", "pubmed"],
        "context_budget": 4000
    }},
    ...
]

Return ONLY the JSON array, no other text."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Research Brief:
{research_brief}

Strategy Mode: {strategy}

Decompose into distinct sub-agent tasks."""
    ),
])


# Error Re-Delegation Prompt (Phase 8.2)
# Required inputs: original_task (str), error_type (str), error_message (str), attempted_tools (str)
RESEARCH_ERROR_RE_DELEGATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research error recovery specialist. A sub-agent task failed and needs to be refined for retry.

Common error patterns and fixes:
- "No results found" → Broaden query, try synonyms, different tools
- "Too many results" → Add more specific constraints, narrow focus
- "Tool unavailable" → Switch to alternative tools
- "Low quality results" → Add quality filters, prefer academic sources

Refine the task to avoid the previous error. Provide:
- Updated task description
- New query variants
- Alternative tools to try
- Any additional context

Return response as JSON:
{{
    "task_id": "<original_id>_retry",
    "description": "Refined task description",
    "query_variants": ["refined query 1", "refined query 2"],
    "preferred_tools": ["alternative tools"],
    "context_budget": 4000,
    "retry_reasoning": "Why this refinement should work"
}}

Return ONLY the JSON object, no other text."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Original Task:
{original_task}

Error Type: {error_type}
Error Message: {error_message}
Attempted Tools: {attempted_tools}

Generate a refined task to avoid this error."""
    ),
])


# Findings Compression Prompt (Phase 8.4)
# Required inputs: raw_results (str), task_description (str)
RESEARCH_FINDINGS_COMPRESSION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research findings compressor. Your job is to extract structured facts from raw search results while preserving key information.

Extract and structure:
- Key facts and claims (3-5 main points)
- Supporting evidence (citations, data)
- Source credibility indicators
- Relationships between findings

Compress into 500-1K tokens while maintaining:
- Accuracy of information
- Citation traceability
- Relevance to task

Return response as JSON:
{{
    "key_facts": ["fact 1", "fact 2", ...],
    "evidence": [
        {{
            "claim": "...",
            "source": "...",
            "credibility_score": 0.0-1.0
        }}
    ],
    "summary": "Brief synthesis of findings"
}}

Return ONLY the JSON object, no other text."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Task: {task_description}

Raw Results:
{raw_results}

Extract and compress key findings."""
    ),
])


# NOTE: Additional prompts for Phase 8 will be added during implementation:
# - Sub-agent query generation
# - Tool selection
# - Gap analysis
# - Aggregation synthesis

