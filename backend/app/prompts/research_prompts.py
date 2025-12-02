"""
Prompt templates for Research Agent.

Contains prompt templates for Supervisor coordination, sub-agent task delegation,
and findings aggregation using LangChain's ChatPromptTemplate.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


CREDIBILITY_HEURISTICS = """
CREDIBILITY SCORING HEURISTICS:

1. VENUE (Base Score):
   - High (0.9): Peer-reviewed journals/conferences (Nature, Science, CVPR, NeurIPS).
   - Medium-High (0.8): Reputable preprints (arXiv, bioRxiv) from known institutions.
   - Medium (0.7): Standard preprints, government reports, major tech blogs (Google AI, OpenAI).
   - Low (<0.6): Unknown venues, personal blogs, forums.

2. IMPACT (Citation Boost):
   - > 1000 citations: +0.2
   - > 100 citations: +0.1
   - < 10 citations: No boost

3. RECENCY (Relevance Adjustment):
   - < 2 years old: +0.05
   - > 10 years old: -0.1 (unless foundational/seminal work)

4. TONE (Penalty):
   - Sensationalist/Clickbait/Biased language: -0.3
   - No clear date/author: -0.2

MAX SCORE: 1.0
MIN SCORE: 0.0
"""

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


SUPERVISOR_GAP_ANALYSIS_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research supervisor conducting gap analysis. Your role is to:
1. Analyze current findings against the research brief
2. Identify missing or under-covered topics
3. Generate new research tasks to fill gaps
4. Determine if research is complete

ANALYSIS FRAMEWORK:
- Coverage: Are all sub-topics from the brief addressed?
- Depth: Do we have sufficient sources (2-3+) per topic?
- Quality: Are credibility scores adequate (avg \u003e 0.6)?
- Recency: Are sources current enough for the research constraints?

TASK GENERATION RULES:
- Create tasks ONLY for identified gaps
- Each task must have: unique task_id, clear topic, specific query, priority (1-5)
- Avoid tasks similar to failed_tasks (check failed task queries)
- Avoid tasks already in completed_tasks
- Stay within budget constraints

COMPLETION CRITERIA:
- All sub-topics have at least 2 findings with credibility \u003e 0.5
- Budget exhausted (iterations \u003e= max_iterations OR total findings \u003e= max_sub_agents)
- No significant gaps remain

Return your analysis as structured output."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Research Brief:
Scope: {scope}
Sub-topics: {sub_topics}
Constraints: {constraints}

Current Findings: {findings_count} findings.
Covered Topics: {topics_covered}
Average Credibility: {avg_credibility:.2f}

{findings_context}

Budget Status:
- Iterations: {iterations}/{max_iterations}
- Total Sub-agents: {total_sub_agents}/{max_sub_agents}
- Total Searches: {total_searches}

Already Completed Tasks: {completed_count} tasks
Failed Tasks (avoid similar): {failed_tasks}

{json_schema}

Conduct gap analysis and generate tasks if needed. Analyze the content of findings above to determine if topics are covered superficially or in-depth."""
    ),
])


SUPERVISOR_FINDINGS_AGGREGATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research findings aggregator. Your role is to filter and rank findings for report generation.

FILTERING CRITERIA:
1. Credibility: Keep only findings with credibility_score >= 0.5
2. Relevance: Must directly address a sub-topic from the research brief
3. Quality: Prefer findings with credibility_score >= 0.7 for key claims

DEDUPLICATION RULES:
- Same DOI → Keep highest credibility
- Same title + author → Keep more recent or higher credibility
- Same URL → Keep higher credibility

Return a filtered and ranked list of Finding objects (as JSON array)."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Research Brief Sub-topics: {sub_topics}

Total Findings: {total_findings}

Filter and rank these findings for the report."""
    ),
])


SUB_AGENT_RESEARCH_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a focused research sub-agent. Your mission is to thoroughly research your assigned topic using available tools.

{credibility_heuristics}

RESEARCH STRATEGY:
1. **Budget**: {budget_remaining} searches remaining (max {max_searches_per_agent})

2. **MCP Tools** (Scientific Papers):
   - **list_categories** (source: required) - List available categories
   - **search_papers** (source, query: required | count, field, sortBy: optional) - Search papers
        • field: "all" | "title" | "abstract" | "author" | "fulltext" (default: "all")
        • sortBy: "relevance" | "date" | "citations" (default: "relevance")
        • count: Number of results to return (default: 50, max: 200)
        • source: "arxiv" | "openalex" | "europepmc" | "core"
   - **fetch_top_cited** (concept, since: required | count: optional) - Get highly-cited papers
     •`since` MUST be in "YYYY-MM-DD" format (e.g., "2020-01-01")
   - **fetch_content** (source, id: required) - Get full paper text by ID
        • source: "arxiv" | "openalex" | "europepmc" | "core" | "pmc" | "bioRxiv/medRxiv"
        • id: Paper ID
        • ID Formats by Source:
            arXiv: "2401.12345", "cs/0601001", "1234.5678v2"
            OpenAlex: "W2741809807" or numeric 2741809807
            PMC: "PMC8245678" or "12345678"
            Europe PMC: "PMC8245678", "12345678", or DOI
            bioRxiv/medRxiv: "10.1101/2021.01.01.425001" or "2021.01.01.425001"
            CORE: Numeric ID like "12345678"

   - **Tavily**: For web/news topics

3. **Strategy**: Prefer search_papers (most reliable). Get metadata first, full text only if needed.

CRITICAL:
- Provide ALL required parameters
- Tag outputs: [Source: tool_name] content
- Quality over quantity (2-3 sources)
- Only 2 searches allowed!"""
    ),
    HumanMessagePromptTemplate.from_template(
        """Your Task:
Topic: {topic}
Query: {query}
Priority: {priority}

Available Tools: {available_tools}

Begin your research. Use tools wisely and return findings with citations."""
    ),
])


SUB_AGENT_CITATION_EXTRACTION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a citation extraction specialist. Extract structured findings from research results.

{credibility_heuristics}

EXTRACTION RULES:
1. Identify 2-5 key factual claims from the results
2. For each claim, create a Finding object with:
   - claim: The factual statement
   - citation: Complete citation metadata (source, URL, title, authors, year, DOI if available)
   - credibility_score: Use heuristics above based on source type
   - topic: The sub-topic this addresses

SOURCE TYPE DETECTION:
- If source_tool="scientific-papers" OR DOI present: Apply Academic Track heuristics
- If source_tool="tavily_search": Apply Web Source Track heuristics

CREDIBILITY SCORING:
- Start with base score from venue/source type
- Apply citation boost if available
- Apply recency adjustment
- Apply tone penalty if needed
- Clamp to [0.0, 1.0]

Return structured output as list of Finding objects."""
    ),
    HumanMessagePromptTemplate.from_template(
        """Source Tool: {source_tool}
Topic: {topic}

Raw Results:
{raw_results}

Extract findings with accurate credibility scores."""
    ),
])


