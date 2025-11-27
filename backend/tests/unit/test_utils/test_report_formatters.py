"""Unit tests for report formatter utilities."""

import pytest
from app.utils.report_formatters import (
    validate_report_format,
    get_default_format,
    extract_comparison_data,
    extract_literature_review_structure,
    extract_gap_analysis_structure
)
from app.models.schemas import Finding, Citation, ReportFormat, SourceType


@pytest.fixture
def sample_findings():
    return [
        Finding(
            claim="AI improves diagnostics",
            citation=Citation(
                source="Journal A",
                url="https://example.com/a",
                title="AI in Medicine",
                authors=["Dr. Smith"],
                year=2023,
                credibility_score=0.9,
                source_type=SourceType.PEER_REVIEWED
            ),
            topic="Healthcare",
            credibility_score=0.9
        ),
        Finding(
            claim="ML enhances treatment",
            citation=Citation(
                source="Journal B",
                url="https://example.com/b",
                title="ML Treatment",
                authors=["Dr. Jones"],
                year=2023,
                credibility_score=0.85,
                source_type=SourceType.PEER_REVIEWED
            ),
            topic="Healthcare",
            credibility_score=0.85
        ),
        Finding(
            claim="AI chatbots improve engagement",
            citation=Citation(
                source="Blog",
                url="https://example.com/blog",
                title="Chatbots",
                credibility_score=0.4,
                source_type=SourceType.BLOG
            ),
            topic="Patient Care",
            credibility_score=0.4
        )
    ]


class TestValidateReportFormat:
    def test_validate_report_format_valid_summary(self):
        result = validate_report_format("summary")
        assert result == ReportFormat.SUMMARY

    def test_validate_report_format_valid_comparison(self):
        result = validate_report_format("comparison")
        assert result == ReportFormat.COMPARISON

    def test_validate_report_format_case_insensitive(self):
        result = validate_report_format("LITERATURE_REVIEW")
        assert result == ReportFormat.LITERATURE_REVIEW

    def test_validate_report_format_whitespace_handling(self):
        result = validate_report_format("  gap_analysis  ")
        assert result == ReportFormat.GAP_ANALYSIS

    def test_validate_report_format_none_returns_other(self):
        result = validate_report_format(None)
        assert result == ReportFormat.OTHER

    def test_validate_report_format_invalid_raises_error(self):
        with pytest.raises(ValueError, match="Invalid format 'invalid'"):
            validate_report_format("invalid")


class TestGetDefaultFormat:
    def test_get_default_format_empty_findings_returns_summary(self):
        result = get_default_format([])
        assert result == ReportFormat.SUMMARY

    def test_get_default_format_many_topics_high_credibility_returns_literature_review(
        self
    ):
        findings = [
            Finding(
                claim=f"Claim {i}",
                citation=Citation(
                    source=f"Source {i}",
                    url=f"http://example.com/{i}",
                    credibility_score=0.8,
                    source_type=SourceType.PEER_REVIEWED
                ),
                topic=f"Topic{i % 5}",
                credibility_score=0.8
            )
            for i in range(15)
        ]
        result = get_default_format(findings)
        assert result == ReportFormat.LITERATURE_REVIEW

    def test_get_default_format_few_topics_returns_comparison(self):
        findings = [
            Finding(
                claim="Claim 1",
                citation=Citation(source="A", url="http://a.com", credibility_score=0.8),
                topic="Topic1",
                credibility_score=0.8
            ),
            Finding(
                claim="Claim 2",
                citation=Citation(source="B", url="http://b.com", credibility_score=0.8),
                topic="Topic2",
                credibility_score=0.8
            ),
            Finding(
                claim="Claim 3",
                citation=Citation(source="C", url="http://c.com", credibility_score=0.8),
                topic="Topic3",
                credibility_score=0.8
            )
        ]
        result = get_default_format(findings)
        assert result == ReportFormat.COMPARISON

    def test_get_default_format_low_credibility_returns_fact_validation(self):
        findings = [
            Finding(
                claim="Claim",
                citation=Citation(source="A", url="http://a.com", credibility_score=0.3),
                topic="Topic",
                credibility_score=0.3
            ),
            Finding(
                claim="Claim 2",
                citation=Citation(source="B", url="http://b.com", credibility_score=0.4),
                topic="Topic",
                credibility_score=0.4
            )
        ]
        result = get_default_format(findings)
        assert result == ReportFormat.FACT_VALIDATION


class TestExtractComparisonData:
    def test_extract_comparison_data_groups_by_topic(self, sample_findings):
        result = extract_comparison_data(sample_findings)
        
        assert "Healthcare" in result
        assert "Patient Care" in result
        assert len(result["Healthcare"]) == 2
        assert len(result["Patient Care"]) == 1

    def test_extract_comparison_data_empty_findings(self):
        result = extract_comparison_data([])
        assert result == {}


class TestExtractLiteratureReviewStructure:
    def test_extract_literature_review_structure_returns_themes_and_stats(
        self, sample_findings
    ):
        result = extract_literature_review_structure(sample_findings)
        
        assert "themes" in result
        assert "statistics" in result
        assert result["statistics"]["total_findings"] == 3
        assert result["statistics"]["theme_count"] == 2

    def test_extract_literature_review_structure_calculates_quality_percentage(
        self, sample_findings
    ):
        result = extract_literature_review_structure(sample_findings)
        
        assert "high_quality_percentage" in result["statistics"]
        assert result["statistics"]["high_quality_percentage"] > 0

    def test_extract_literature_review_structure_counts_peer_reviewed(
        self, sample_findings
    ):
        result = extract_literature_review_structure(sample_findings)
        
        assert result["statistics"]["peer_reviewed_count"] == 2


class TestExtractGapAnalysisStructure:
    def test_extract_gap_analysis_structure_identifies_well_covered(self):
        findings = [
            Finding(
                claim=f"Claim {i}",
                citation=Citation(
                    source=f"Source {i}",
                    url=f"http://example.com/{i}",
                    credibility_score=0.8
                ),
                topic="Well Covered",
                credibility_score=0.8
            )
            for i in range(5)
        ]
        
        result = extract_gap_analysis_structure(findings)
        
        assert len(result["well_covered_topics"]) == 1
        assert result["well_covered_topics"][0]["topic"] == "Well Covered"

    def test_extract_gap_analysis_structure_identifies_under_researched(self):
        finding = Finding(
            claim="Claim",
            citation=Citation(source="A", url="http://a.com", credibility_score=0.8),
            topic="Under Researched",
            credibility_score=0.8
        )
        
        result = extract_gap_analysis_structure([finding])
        
        assert len(result["under_researched_topics"]) == 1

    def test_extract_gap_analysis_structure_identifies_low_quality(self):
        findings = [
            Finding(
                claim=f"Claim {i}",
                citation=Citation(
                    source=f"Source {i}",
                    url=f"http://example.com/{i}",
                    credibility_score=0.3
                ),
                topic="Low Quality",
                credibility_score=0.3
            )
            for i in range(3)
        ]
        
        result = extract_gap_analysis_structure(findings)
        
        assert len(result["low_quality_topics"]) == 1


def test_get_credibility_warning():
    from app.utils.report_formatters import get_credibility_warning
    
    assert get_credibility_warning(0.4) == "⚠️ Low Credibility Source"
    assert get_credibility_warning(0.49) == "⚠️ Low Credibility Source"
    assert get_credibility_warning(0.5) is None
    assert get_credibility_warning(0.9) is None
