"""Scope Agent - Human-in-the-loop clarification agent.

This agent handles multi-turn clarification conversations with users to determine
the research scope. It asks clarifying questions and generates a research brief
once the scope is sufficient.

Future implementation:
- async def clarify_scope(user_query: str, conversation_history: Optional[List], previous_questions: Optional[List]) -> Union[ClarificationQuestions, ResearchBrief]
- Multi-turn conversation logic
- Completion detection (decide when scope is sufficient)
- Research brief generation from clarification conversation
"""

