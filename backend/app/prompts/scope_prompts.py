"""
Prompt templates for Scope Agent.

Contains prompt templates for multi-turn clarification conversations and
research brief generation using LangChain's ChatPromptTemplate.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


SCOPE_QUESTION_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research scope clarification assistant. Your job is to ask clear, concise clarifying questions to understand the user's research needs better.

Your goal is to gather information about:
- The specific aspect or angle they want to research
- The depth and breadth of research needed
- Any constraints (time period, geographic focus, etc.)
- The intended use or audience for the research
- Preferred format for the final report (if not already clear)

Ask straightforward questions without explaining why you're asking them. The context field should provide overall reasoning.

If the scope seems sufficiently clear, return an empty list of questions.

{format_instructions}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """User's Query: {user_query}

Conversation History:
{conversation_history}

Based on the user's query and conversation so far, generate 1-3 clarifying questions."""
    ),
])


SCOPE_COMPLETION_DETECTION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research scope analyzer. Your job is to determine if we have enough information to proceed with research.

Analyze if we have enough information to:
1. Understand what the user wants to research
2. Know the scope and boundaries of the research
3. Understand any constraints or requirements
4. Know what format the final deliverable should take

{format_instructions}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """User's Original Query: {user_query}

Conversation History:
{conversation_history}

Determine if we have sufficient information to proceed with research."""
    ),
])


SCOPE_BRIEF_GENERATION_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a research brief generator. Based on the clarified conversation, create a structured research brief.

The research brief should include:
- Clear research scope and boundaries
- List of specific sub_topics to investigate
- Any constraints (time periods, source types, depth level, etc.)
- Expected deliverables and format (summary, comparison, ranking, literature review, gap analysis, etc.)
- Any other relevant metadata

{format_instructions}"""
    ),
    HumanMessagePromptTemplate.from_template(
        """User's Original Query: {user_query}

Conversation History:
{conversation_history}

Generate a comprehensive research brief based on the conversation."""
    ),
])

