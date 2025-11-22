"""Agent behavior evaluation tests using G-Eval.

This module tests agent behavior using DeepEval's G-Eval metric with custom criteria:
- Format compliance: Validates output structure matches requirements
- Tool selection accuracy: Validates correct tool usage for queries
- Reasoning validity: Evaluates quality of agent reasoning steps

G-Eval uses LLMs to evaluate outputs against custom criteria, enabling flexible
behavioral testing beyond standard metrics.
"""

import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval


class TestFormatCompliance:
    """Test suite for format compliance using G-Eval.
    
    Format compliance ensures outputs follow required structure and formatting.
    Threshold: > 0.8 (80% compliance required)
    """
    
    def test_research_brief_format_compliance(self, gemini_evaluation_model):
        """Test that research briefs follow the required format structure.
        
        Required format:
        - Scope: Clear research scope statement
        - Sub-topics: List of specific sub-topics
        - Constraints: Dictionary of constraints (time_range, source_types, etc.)
        - Format type: Desired output format
        - Depth level: Research depth (basic, detailed, comprehensive)
        """
        # Define the format compliance metric
        format_metric = GEval(
            name="Research Brief Format Compliance",
            model=gemini_evaluation_model,
            criteria=(
                "Does the research brief contain all required sections: "
                "Scope, Sub-topics, Constraints, Format type, and Depth level? "
                "Are the sections properly structured and complete?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input="Generate a research brief for AI in healthcare diagnostics",
            actual_output=(
                "Research Brief:\n\n"
                "Scope: AI applications in medical diagnostics\n\n"
                "Sub-topics:\n"
                "- Deep learning for medical imaging\n"
                "- Natural language processing for clinical notes\n"
                "- Predictive models for disease diagnosis\n\n"
                "Constraints:\n"
                "- Time range: 2019-2024\n"
                "- Source types: peer-reviewed, academic\n"
                "- Minimum credibility: 0.7\n\n"
                "Format type: literature_review\n\n"
                "Depth level: detailed"
            ),
        )
        
        # Measure format compliance
        format_metric.measure(test_case)
        
        assert format_metric.score >= 0.8, (
            f"Research brief format compliance score {format_metric.score} is below threshold. "
            f"Reason: {format_metric.reason}"
        )
    
    def test_gap_analysis_format_compliance(self, gemini_evaluation_model):
        """Test that gap analysis reports follow the required format.
        
        Required format:
        - Executive Summary
        - Coverage Analysis (with metrics and visual indicators)
        - Gap Identification (categorized by type)
        - Recommendations
        - Bibliography (with credibility indicators)
        """
        format_metric = GEval(
            name="Gap Analysis Format Compliance",
            model=gemini_evaluation_model,
            criteria=(
                "Does the gap analysis report strictly follow the required structure: "
                "Executive Summary, Coverage Analysis, Gap Identification, "
                "Recommendations, and Bibliography? "
                "Are coverage metrics and gap categories properly included?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input="Generate a gap analysis report for the research findings",
            actual_output=(
                "# Gap Analysis Report\n\n"
                "## Executive Summary\n"
                "Research covers 3 main topics with 15 sources. "
                "Key gaps identified in real-time diagnostics and edge deployment.\n\n"
                "## Coverage Analysis\n"
                "- Medical imaging: [Well-covered] (8 sources, avg credibility: 0.85)\n"
                "- Clinical NLP: [Well-covered] (5 sources, avg credibility: 0.80)\n"
                "- Real-time diagnostics: [Under-researched] (2 sources, avg credibility: 0.70)\n\n"
                "## Gap Identification\n"
                "**Coverage Gaps:**\n"
                "- Limited research on edge deployment\n\n"
                "**Depth Gaps:**\n"
                "- Real-time diagnostics needs more investigation\n\n"
                "## Recommendations\n"
                "1. Focus future research on edge deployment strategies\n"
                "2. Investigate real-time diagnostic systems\n\n"
                "## Bibliography\n"
                "[1] Smith et al. (2023) - Deep Learning for Medical Imaging [Credibility: 0.92]"
            ),
        )
        
        format_metric.measure(test_case)
        
        assert format_metric.score >= 0.8, (
            f"Gap analysis format compliance score {format_metric.score} is below threshold. "
            f"Reason: {format_metric.reason}"
        )
    
    def test_literature_review_format_compliance(self, gemini_evaluation_model):
        """Test that literature reviews follow the required format.
        
        Required format:
        - Introduction
        - Thematic Sections (organized by sub-topic)
        - Research Gaps
        - Conclusion
        - Bibliography (with credibility indicators)
        """
        format_metric = GEval(
            name="Literature Review Format Compliance",
            model=gemini_evaluation_model,
            criteria=(
                "Does the literature review follow the required structure: "
                "Introduction, Thematic Sections, Research Gaps, Conclusion, "
                "and Bibliography? Are sources properly cited with credibility indicators?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input="Generate a literature review for AI in medical diagnostics",
            actual_output=(
                "# Literature Review: AI in Medical Diagnostics\n\n"
                "## Introduction\n"
                "This review examines recent advances in AI for medical diagnostics.\n\n"
                "## Deep Learning for Medical Imaging\n"
                "Recent studies show 95% accuracy in lung cancer detection [1].\n\n"
                "## NLP for Clinical Notes\n"
                "NLP models extract clinical information with 88% precision [2].\n\n"
                "## Research Gaps\n"
                "Limited research on edge deployment and real-time systems.\n\n"
                "## Conclusion\n"
                "AI shows promise but gaps remain in deployment strategies.\n\n"
                "## Bibliography\n"
                "[1] Smith et al. (2023) [Credibility: 0.92]\n"
                "[2] Doe et al. (2022) [Credibility: 0.85]"
            ),
        )
        
        format_metric.measure(test_case)
        
        assert format_metric.score >= 0.8, (
            f"Literature review format compliance score {format_metric.score} is below threshold. "
            f"Reason: {format_metric.reason}"
        )


class TestToolSelectionAccuracy:
    """Test suite for tool selection accuracy using G-Eval.
    
    Tool selection accuracy ensures agents choose appropriate tools for queries.
    """
    
    def test_academic_query_tool_selection(self, gemini_evaluation_model):
        """Test that academic queries use academic tools (not general web search).
        
        Academic queries should prefer:
        - Scientific Paper Harvester (arXiv, PubMed, OpenAlex, etc.)
        - Academic databases
        
        Not:
        - General web search (Tavily)
        - Blog/news sources
        """
        tool_selection_metric = GEval(
            name="Academic Tool Selection",
            model=gemini_evaluation_model,
            criteria=(
                "Did the agent select academic tools (Scientific Paper Harvester, "
                "arXiv, PubMed, OpenAlex) for an academic research query? "
                "Academic queries should NOT use general web search tools."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input="Find recent papers on CRISPR gene editing",
            actual_output=(
                "Tool selection:\n"
                "- Selected: Scientific Paper Harvester (search_papers)\n"
                "- Sources: PubMed Central, arXiv, OpenAlex\n"
                "- Reason: Academic query requires peer-reviewed sources"
            ),
        )
        
        tool_selection_metric.measure(test_case)
        
        assert tool_selection_metric.score >= 0.8, (
            f"Academic tool selection score {tool_selection_metric.score} is below threshold. "
            f"Reason: {tool_selection_metric.reason}"
        )
    
    def test_general_query_tool_selection(self, gemini_evaluation_model):
        """Test that general queries use appropriate web search tools.
        
        General queries should use:
        - Tavily web search
        - General search engines
        
        Not necessarily:
        - Academic-only tools
        """
        tool_selection_metric = GEval(
            name="General Query Tool Selection",
            model=gemini_evaluation_model,
            criteria=(
                "Did the agent select appropriate tools for a general web search query? "
                "General queries can use Tavily or other web search tools."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input="What are the latest trends in AI startups?",
            actual_output=(
                "Tool selection:\n"
                "- Selected: Tavily web search\n"
                "- Reason: General query about industry trends, not academic research"
            ),
        )
        
        tool_selection_metric.measure(test_case)
        
        assert tool_selection_metric.score >= 0.8, (
            f"General query tool selection score {tool_selection_metric.score} is below threshold. "
            f"Reason: {tool_selection_metric.reason}"
        )
    
    def test_mixed_query_tool_selection(self, gemini_evaluation_model):
        """Test that mixed queries use both academic and general tools appropriately.
        
        Mixed queries (academic + industry context) should use:
        - Academic tools for research findings
        - Web search for industry context
        """
        tool_selection_metric = GEval(
            name="Mixed Query Tool Selection",
            model=gemini_evaluation_model,
            criteria=(
                "Did the agent select both academic tools (for research) and "
                "web search tools (for industry context) for a mixed query? "
                "Mixed queries require balanced tool selection."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input="Compare academic research on AI safety with industry implementations",
            actual_output=(
                "Tool selection:\n"
                "- Selected: Scientific Paper Harvester (for academic research)\n"
                "- Selected: Tavily web search (for industry implementations)\n"
                "- Reason: Mixed query requires both academic and industry sources"
            ),
        )
        
        tool_selection_metric.measure(test_case)
        
        assert tool_selection_metric.score >= 0.8, (
            f"Mixed query tool selection score {tool_selection_metric.score} is below threshold. "
            f"Reason: {tool_selection_metric.reason}"
        )


class TestReasoningValidity:
    """Test suite for reasoning validity using G-Eval.
    
    Reasoning validity evaluates the quality of agent reasoning and decision-making.
    """
    
    def test_clarification_reasoning(self, gemini_evaluation_model):
        """Test that clarification question reasoning is valid and helpful.
        
        Valid reasoning should:
        - Identify ambiguity in user query
        - Explain why clarification is needed
        - Ask specific, actionable questions
        """
        reasoning_metric = GEval(
            name="Clarification Reasoning Validity",
            model=gemini_evaluation_model,
            criteria=(
                "Is the agent's reasoning for asking clarification questions valid? "
                "Does it identify specific ambiguities and ask actionable questions?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input="User query: I need to research AI",
            actual_output=(
                "Reasoning: The query 'AI' is too broad and ambiguous. "
                "Need to clarify:\n"
                "1. Specific AI domain (ML, NLP, CV, robotics, etc.)\n"
                "2. Application area (healthcare, finance, education, etc.)\n"
                "3. Time period of interest\n"
                "4. Desired output format\n\n"
                "Clarification question: What specific aspect of AI are you interested in? "
                "(e.g., machine learning, natural language processing, computer vision)"
            ),
        )
        
        reasoning_metric.measure(test_case)
        
        assert reasoning_metric.score >= 0.8, (
            f"Clarification reasoning score {reasoning_metric.score} is below threshold. "
            f"Reason: {reasoning_metric.reason}"
        )
    
    def test_research_strategy_reasoning(self, gemini_evaluation_model):
        """Test that research strategy selection reasoning is valid.
        
        Valid reasoning should:
        - Analyze query complexity
        - Justify strategy choice (FLAT, FLAT_REFINEMENT, HIERARCHICAL)
        - Explain expected sub-agent count and depth
        """
        reasoning_metric = GEval(
            name="Research Strategy Reasoning Validity",
            model=gemini_evaluation_model,
            criteria=(
                "Is the agent's reasoning for selecting a research strategy valid? "
                "Does it analyze query complexity and justify the strategy choice?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input=(
                "Research Brief: AI in medical diagnostics\n"
                "Sub-topics: 3 topics (deep learning, NLP, predictive models)\n"
                "Depth: detailed"
            ),
            actual_output=(
                "Strategy Selection: FLAT_REFINEMENT\n\n"
                "Reasoning:\n"
                "- Query complexity: Medium (3 sub-topics, detailed depth)\n"
                "- Sub-topics are distinct and non-overlapping\n"
                "- FLAT strategy insufficient (may miss depth)\n"
                "- HIERARCHICAL unnecessary (not complex enough)\n"
                "- FLAT_REFINEMENT appropriate: initial research + gap-based refinement\n\n"
                "Expected: 3-5 sub-agents, 1 refinement round if gaps detected"
            ),
        )
        
        reasoning_metric.measure(test_case)
        
        assert reasoning_metric.score >= 0.8, (
            f"Research strategy reasoning score {reasoning_metric.score} is below threshold. "
            f"Reason: {reasoning_metric.reason}"
        )
    
    def test_gap_detection_reasoning(self, gemini_evaluation_model):
        """Test that gap detection reasoning is valid and actionable.
        
        Valid reasoning should:
        - Identify specific gaps (coverage, depth, temporal, etc.)
        - Explain why gaps are significant
        - Provide actionable recommendations
        """
        reasoning_metric = GEval(
            name="Gap Detection Reasoning Validity",
            model=gemini_evaluation_model,
            criteria=(
                "Is the agent's reasoning for identifying research gaps valid? "
                "Does it identify specific gaps and provide actionable recommendations?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input=(
                "Research Brief: AI in medical diagnostics (3 sub-topics)\n"
                "Findings: 15 sources total\n"
                "- Medical imaging: 8 sources (avg credibility: 0.85)\n"
                "- Clinical NLP: 5 sources (avg credibility: 0.80)\n"
                "- Real-time diagnostics: 2 sources (avg credibility: 0.70)"
            ),
            actual_output=(
                "Gap Analysis:\n\n"
                "**Coverage Gap Detected:**\n"
                "- Real-time diagnostics: Only 2 sources (threshold: 3 minimum)\n"
                "- Reasoning: Insufficient coverage for comprehensive analysis\n\n"
                "**Quality Gap Detected:**\n"
                "- Real-time diagnostics: Average credibility 0.70 (threshold: 0.75)\n"
                "- Reasoning: Below quality threshold, need higher-credibility sources\n\n"
                "**Recommendation:**\n"
                "- Trigger targeted re-research for real-time diagnostics\n"
                "- Prefer academic sources (PubMed, arXiv) for higher credibility\n"
                "- Expand search to recent papers (last 2 years)"
            ),
        )
        
        reasoning_metric.measure(test_case)
        
        assert reasoning_metric.score >= 0.8, (
            f"Gap detection reasoning score {reasoning_metric.score} is below threshold. "
            f"Reason: {reasoning_metric.reason}"
        )


class TestAgentBehaviorIntegration:
    """Integration tests combining multiple G-Eval behavioral metrics."""
    
    def test_complete_agent_behavior_evaluation(self, gemini_evaluation_model):
        """Test multiple behavioral dimensions simultaneously.
        
        Evaluates format compliance, tool selection, and reasoning together.
        """
        # Define metrics
        format_metric = GEval(
            name="Format Compliance",
            model=gemini_evaluation_model,
            criteria="Does the output follow the required format structure?",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        tool_metric = GEval(
            name="Tool Selection",
            model=gemini_evaluation_model,
            criteria="Did the agent select appropriate tools for the query?",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        reasoning_metric = GEval(
            name="Reasoning Validity",
            model=gemini_evaluation_model,
            criteria="Is the agent's reasoning valid and well-justified?",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        )
        
        test_case = LLMTestCase(
            input="Find recent papers on CRISPR and generate a literature review",
            actual_output=(
                "Tool Selection:\n"
                "- Selected: Scientific Paper Harvester (search_papers)\n"
                "- Reasoning: Academic query requires peer-reviewed sources\n\n"
                "Research Brief:\n"
                "Scope: CRISPR gene editing research\n"
                "Sub-topics: CRISPR-Cas9, Off-target effects, Clinical applications\n"
                "Format: literature_review\n\n"
                "Output Format:\n"
                "- Introduction\n"
                "- Thematic Sections\n"
                "- Research Gaps\n"
                "- Conclusion\n"
                "- Bibliography"
            ),
        )
        
        # Evaluate all metrics
        evaluate(
            test_cases=[test_case],
            metrics=[format_metric, tool_metric, reasoning_metric],
        )
        
        # Assert all metrics pass
        assert format_metric.score >= 0.8, (
            f"Format compliance failed: {format_metric.reason}"
        )
        assert tool_metric.score >= 0.8, (
            f"Tool selection failed: {tool_metric.reason}"
        )
        assert reasoning_metric.score >= 0.8, (
            f"Reasoning validity failed: {reasoning_metric.reason}"
        )


