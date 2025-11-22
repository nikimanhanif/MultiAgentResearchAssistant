"""Agent orchestration logic for pipeline coordination.

This module coordinates the 3-agent pipeline flow:
1. Scope Agent (multi-turn clarification → research brief)
2. Research Agent (uses brief, coordinates via LangGraph → summarized findings)
3. Report Agent (one-shot, takes brief + findings → markdown report)

Future implementation:
- Pipeline orchestration function
- Conversation state management
- Phase transitions (CLARIFYING → RESEARCHING → REPORTING → COMPLETE)
- Coordination between functional agents and LangGraph
"""

