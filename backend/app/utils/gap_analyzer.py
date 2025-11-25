"""Research gap detection and analysis.

Updated in Phase 3.7: Gap analysis is now LLM-based (supervisor prompt), not algorithmic.

The Supervisor Agent uses LLM to analyze gaps in findings vs research brief.
No programmatic set-difference or rule-based gap detection.

Implementation: Phase 8.5

Future implementation:
- Supervisor LLM performs gap analysis via prompt
- Filters findings based on relevance and credibility
- Programmatic deduplication (by DOI, title+author, URL)
- Programmatic sorting (by credibility score)
"""

# TODO: Phase 8.5 - Implement detect_gaps_algorithmic() for zero-token gap detection
# TODO: Phase 8.5 - Implement should_trigger_re_research() for conditional re-research
# TODO: Phase 8.5 - Implement create_targeted_re_research_tasks() for gap-based re-research
