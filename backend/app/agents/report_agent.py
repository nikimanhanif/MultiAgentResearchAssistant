"""Report Agent - One-shot markdown report generation.

This agent takes the research brief and list of findings to generate
a final markdown-formatted report for the user.

Architecture: Supervisor Loop (Phase 3.7+)
- Input: ResearchBrief + List[Finding] (each with embedded Citation)
- Output: Markdown report with citations

Future implementation:
- async def generate_report(research_brief: ResearchBrief, findings: List[Finding]) -> str
- One-shot report generation (no sub-agents, no iterative refinement)
- Markdown formatting
- Citation handling: Use finding index [0], [1], [2]... for in-text citations
- Report formatting based on brief specifications
"""

