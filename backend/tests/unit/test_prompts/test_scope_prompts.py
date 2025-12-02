"""Unit tests for Scope Agent prompt templates."""

import pytest
from langchain_core.prompts import ChatPromptTemplate

from app.prompts.scope_prompts import (
    SCOPE_QUESTION_GENERATION_TEMPLATE,
    SCOPE_COMPLETION_DETECTION_TEMPLATE,
    SCOPE_BRIEF_GENERATION_TEMPLATE,
)


class TestScopeQuestionGenerationTemplate:
    """Test cases for SCOPE_QUESTION_GENERATION_TEMPLATE."""

    def test_template_is_chat_prompt_template(self):
        """Test that template is a ChatPromptTemplate instance."""
        assert isinstance(SCOPE_QUESTION_GENERATION_TEMPLATE, ChatPromptTemplate)

    def test_template_has_required_input_variables(self):
        """Test that template contains required input variables."""
        input_vars = SCOPE_QUESTION_GENERATION_TEMPLATE.input_variables
        assert "user_query" in input_vars
        assert "conversation_history" in input_vars
        assert "format_instructions" in input_vars

    def test_template_has_exactly_two_input_variables(self):
        """Test that template has exactly three input variables."""
        input_vars = SCOPE_QUESTION_GENERATION_TEMPLATE.input_variables
        assert len(input_vars) == 3

    def test_template_formats_with_valid_inputs(self):
        """Test that template formats correctly with valid inputs."""
        formatted = SCOPE_QUESTION_GENERATION_TEMPLATE.format_messages(
            user_query="What is machine learning?",
            conversation_history="No previous conversation.",
            format_instructions="Test format instructions"
        )
        assert len(formatted) == 2  # System + Human messages
        assert formatted[0].type == "system"
        assert formatted[1].type == "human"

    def test_template_system_message_contains_instructions(self):
        """Test that system message contains clarification instructions."""
        formatted = SCOPE_QUESTION_GENERATION_TEMPLATE.format_messages(
            user_query="Test query",
            conversation_history="Test history",
            format_instructions="Test format instructions"
        )
        system_msg = formatted[0].content
        assert "clarification" in system_msg.lower()
        assert "questions" in system_msg.lower()

    def test_template_human_message_contains_query(self):
        """Test that human message contains the user query."""
        test_query = "What is deep learning?"
        formatted = SCOPE_QUESTION_GENERATION_TEMPLATE.format_messages(
            user_query=test_query,
            conversation_history="No conversation",
            format_instructions="Test format instructions"
        )
        human_msg = formatted[1].content
        assert test_query in human_msg

    def test_template_human_message_contains_history(self):
        """Test that human message contains conversation history."""
        test_history = "USER: Previous question\nASSISTANT: Previous answer"
        formatted = SCOPE_QUESTION_GENERATION_TEMPLATE.format_messages(
            user_query="Test query",
            conversation_history=test_history,
            format_instructions="Test format instructions"
        )
        human_msg = formatted[1].content
        assert test_history in human_msg

    def test_template_with_empty_strings(self):
        """Test that template handles empty strings."""
        formatted = SCOPE_QUESTION_GENERATION_TEMPLATE.format_messages(
            user_query="",
            conversation_history="",
            format_instructions="Test format instructions"
        )
        assert len(formatted) == 2
        # Should not raise errors with empty strings

    def test_template_with_long_inputs(self):
        """Test that template handles long inputs."""
        long_query = "What is " * 1000  # Very long query
        long_history = "Turn: " * 1000  # Very long history
        formatted = SCOPE_QUESTION_GENERATION_TEMPLATE.format_messages(
            user_query=long_query,
            conversation_history=long_history,
            format_instructions="Test format instructions"
        )
        assert len(formatted) == 2


class TestScopeCompletionDetectionTemplate:
    """Test cases for SCOPE_COMPLETION_DETECTION_TEMPLATE."""

    def test_template_is_chat_prompt_template(self):
        """Test that template is a ChatPromptTemplate instance."""
        assert isinstance(SCOPE_COMPLETION_DETECTION_TEMPLATE, ChatPromptTemplate)

    def test_template_has_required_input_variables(self):
        """Test that template contains required input variables."""
        input_vars = SCOPE_COMPLETION_DETECTION_TEMPLATE.input_variables
        assert "user_query" in input_vars
        assert "conversation_history" in input_vars
        assert "format_instructions" in input_vars

    def test_template_has_exactly_two_input_variables(self):
        """Test that template has exactly three input variables."""
        input_vars = SCOPE_COMPLETION_DETECTION_TEMPLATE.input_variables
        assert len(input_vars) == 3

    def test_template_formats_with_valid_inputs(self):
        """Test that template formats correctly with valid inputs."""
        formatted = SCOPE_COMPLETION_DETECTION_TEMPLATE.format_messages(
            user_query="What is AI?",
            conversation_history="USER: Question\nASSISTANT: Answer",
            format_instructions="Test format instructions"
        )
        assert len(formatted) == 2  # System + Human messages
        assert formatted[0].type == "system"
        assert formatted[1].type == "human"


    def test_template_system_message_contains_analyzer_role(self):
        """Test that system message defines analyzer role."""
        formatted = SCOPE_COMPLETION_DETECTION_TEMPLATE.format_messages(
            user_query="Test query",
            conversation_history="Test history",
            format_instructions="Test format instructions"
        )
        system_msg = formatted[0].content
        assert "analyzer" in system_msg.lower()
        assert "scope" in system_msg.lower()

    def test_template_mentions_required_analysis_criteria(self):
        """Test that template mentions completion criteria."""
        formatted = SCOPE_COMPLETION_DETECTION_TEMPLATE.format_messages(
            user_query="Test",
            conversation_history="Test",
            format_instructions="Test format instructions"
        )
        system_msg = formatted[0].content
        # Check for key analysis criteria
        assert any(term in system_msg.lower() for term in ["scope", "boundaries", "constraints"])

    def test_template_human_message_structure(self):
        """Test that human message has correct structure."""
        formatted = SCOPE_COMPLETION_DETECTION_TEMPLATE.format_messages(
            user_query="Test query",
            conversation_history="Test history",
            format_instructions="Test format instructions"
        )
        human_msg = formatted[1].content
        assert "Test query" in human_msg
        assert "Test history" in human_msg

    def test_template_with_empty_history(self):
        """Test that template handles empty conversation history."""
        formatted = SCOPE_COMPLETION_DETECTION_TEMPLATE.format_messages(
            user_query="Initial query",
            conversation_history="",
            format_instructions="Test format instructions"
        )
        assert len(formatted) == 2
        human_msg = formatted[1].content
        assert "Initial query" in human_msg


class TestScopeBriefGenerationTemplate:
    """Test cases for SCOPE_BRIEF_GENERATION_TEMPLATE."""

    def test_template_is_chat_prompt_template(self):
        """Test that template is a ChatPromptTemplate instance."""
        assert isinstance(SCOPE_BRIEF_GENERATION_TEMPLATE, ChatPromptTemplate)

    def test_template_has_required_input_variables(self):
        """Test that template contains required input variables."""
        input_vars = SCOPE_BRIEF_GENERATION_TEMPLATE.input_variables
        assert "user_query" in input_vars
        assert "conversation_history" in input_vars
        assert "format_instructions" in input_vars

    def test_template_has_exactly_two_input_variables(self):
        """Test that template has exactly three input variables."""
        input_vars = SCOPE_BRIEF_GENERATION_TEMPLATE.input_variables
        assert len(input_vars) == 3

    def test_template_formats_with_valid_inputs(self):
        """Test that template formats correctly with valid inputs."""
        formatted = SCOPE_BRIEF_GENERATION_TEMPLATE.format_messages(
            user_query="Research topic",
            conversation_history="Q: Question\nA: Answer",
            format_instructions="Test format instructions"
        )
        assert len(formatted) == 2  # System + Human messages
        assert formatted[0].type == "system"
        assert formatted[1].type == "human"

    def test_template_system_message_defines_generator_role(self):
        """Test that system message defines research brief generator role."""
        formatted = SCOPE_BRIEF_GENERATION_TEMPLATE.format_messages(
            user_query="Test",
            conversation_history="Test",
            format_instructions="Test format instructions"
        )
        system_msg = formatted[0].content
        assert "research brief" in system_msg.lower()
        assert "generator" in system_msg.lower()

    def test_template_mentions_required_brief_components(self):
        """Test that template mentions all required brief components."""
        formatted = SCOPE_BRIEF_GENERATION_TEMPLATE.format_messages(
            user_query="Test",
            conversation_history="Test",
            format_instructions="Test format instructions"
        )
        system_msg = formatted[0].content
        # Check for key brief components
        required_components = ["scope", "sub_topics", "constraints", "deliverables", "format"]
        for component in required_components:
            assert component in system_msg.lower()

    def test_template_includes_format_options(self):
        """Test that template includes report format options."""
        formatted = SCOPE_BRIEF_GENERATION_TEMPLATE.format_messages(
            user_query="Test",
            conversation_history="Test",
            format_instructions="Test format instructions"
        )
        system_msg = formatted[0].content
        # Should mention various format types
        assert any(fmt in system_msg.lower() for fmt in ["summary", "comparison", "ranking", "literature", "gap"])

    def test_template_human_message_includes_query_and_history(self):
        """Test that human message includes both query and history."""
        test_query = "Research machine learning"
        test_history = "USER: Previous question\nASSISTANT: Previous answer"
        formatted = SCOPE_BRIEF_GENERATION_TEMPLATE.format_messages(
            user_query=test_query,
            conversation_history=test_history,
            format_instructions="Test format instructions"
        )
        human_msg = formatted[1].content
        assert test_query in human_msg
        assert test_history in human_msg

    def test_template_with_complex_conversation_history(self):
        """Test that template handles complex multi-turn conversations."""
        complex_history = """USER: What is machine learning?
ASSISTANT: Machine learning is a subset of AI.
USER: What types exist?
ASSISTANT: Supervised, unsupervised, and reinforcement learning.
USER: I want to research supervised learning in healthcare."""
        formatted = SCOPE_BRIEF_GENERATION_TEMPLATE.format_messages(
            user_query="Machine learning research",
            conversation_history=complex_history,
            format_instructions="Test format instructions"
        )
        human_msg = formatted[1].content
        assert "supervised learning" in human_msg
        assert "healthcare" in human_msg


class TestScopePromptsIntegration:
    """Integration tests for scope prompts module."""

    def test_all_templates_are_distinct(self):
        """Test that all three templates are different objects."""
        assert SCOPE_QUESTION_GENERATION_TEMPLATE is not SCOPE_COMPLETION_DETECTION_TEMPLATE
        assert SCOPE_QUESTION_GENERATION_TEMPLATE is not SCOPE_BRIEF_GENERATION_TEMPLATE
        assert SCOPE_COMPLETION_DETECTION_TEMPLATE is not SCOPE_BRIEF_GENERATION_TEMPLATE

    def test_all_templates_use_same_input_variables(self):
        """Test that all templates expect the same input variables for consistency."""
        question_vars = set(SCOPE_QUESTION_GENERATION_TEMPLATE.input_variables)
        completion_vars = set(SCOPE_COMPLETION_DETECTION_TEMPLATE.input_variables)
        brief_vars = set(SCOPE_BRIEF_GENERATION_TEMPLATE.input_variables)
        
        assert question_vars == completion_vars == brief_vars

    def test_templates_can_be_used_in_sequence(self):
        """Test that templates can be used sequentially in a workflow."""
        test_query = "Research AI applications"
        test_history = "USER: Tell me more\nASSISTANT: Clarifying..."
        
        # Should be able to format all three in sequence
        questions = SCOPE_QUESTION_GENERATION_TEMPLATE.format_messages(
            user_query=test_query,
            conversation_history=test_history,
            format_instructions="Test format instructions"
        )
        completion = SCOPE_COMPLETION_DETECTION_TEMPLATE.format_messages(
            user_query=test_query,
            conversation_history=test_history,
            format_instructions="Test format instructions"
        )
        brief = SCOPE_BRIEF_GENERATION_TEMPLATE.format_messages(
            user_query=test_query,
            conversation_history=test_history,
            format_instructions="Test format instructions"
        )
        
        assert len(questions) == 2
        assert len(completion) == 2
        assert len(brief) == 2

    def test_templates_with_special_characters(self):
        """Test that templates handle special characters correctly."""
        special_query = "What is AI? (including ML & DL)"
        special_history = 'USER: "Quote test"\nASSISTANT: \'Single quote\''
        
        for template in [
            SCOPE_QUESTION_GENERATION_TEMPLATE,
            SCOPE_COMPLETION_DETECTION_TEMPLATE,
            SCOPE_BRIEF_GENERATION_TEMPLATE
        ]:
            formatted = template.format_messages(
                user_query=special_query,
                conversation_history=special_history,
                format_instructions="Test format instructions"
            )
            assert len(formatted) == 2
            # Special characters should be preserved in messages
            combined_content = formatted[0].content + formatted[1].content
            assert "?" in combined_content or "AI" in combined_content

    def test_templates_with_unicode_characters(self):
        """Test that templates handle unicode characters."""
        unicode_query = "机器学习是什么？"  # Chinese
        unicode_history = "USER: Émile\nASSISTANT: Müller"  # Accented characters
        
        for template in [
            SCOPE_QUESTION_GENERATION_TEMPLATE,
            SCOPE_COMPLETION_DETECTION_TEMPLATE,
            SCOPE_BRIEF_GENERATION_TEMPLATE
        ]:
            formatted = template.format_messages(
                user_query=unicode_query,
                conversation_history=unicode_history,
                format_instructions="Test format instructions"
            )
            assert len(formatted) == 2
            # Should not raise encoding errors

