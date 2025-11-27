"""Report formatting utilities for validation and data extraction."""

from typing import List, Dict, Any, Optional
from app.models.schemas import Finding, ReportFormat


def validate_report_format(format_str: Optional[str]) -> ReportFormat:
    """Validate and convert string to ReportFormat enum.
    
    Args:
        format_str: Format string (case-insensitive)
        
    Returns:
        ReportFormat enum value
        
    Raises:
        ValueError: If format string is invalid
    """
    if format_str is None:
        return ReportFormat.OTHER
    
    format_lower = format_str.lower().strip()
   
    try:
        return ReportFormat(format_lower)
    except ValueError:
        valid_formats = [f.value for f in ReportFormat]
        raise ValueError(
            f"Invalid format '{format_str}'. Must be one of: {', '.join(valid_formats)}"
        )


def get_default_format(findings: List[Finding]) -> ReportFormat:
    """Determine default report format based on findings characteristics.
    
    Args:
        findings: List of Finding objects
        
    Returns:
        Recommended ReportFormat
    """
    if not findings:
        return ReportFormat.SUMMARY
    
    num_findings = len(findings)
    unique_topics = len(set(f.topic for f in findings))
    avg_credibility = sum(f.credibility_score for f in findings) / num_findings
    
    if num_findings >= 10 and unique_topics >= 4:
        if avg_credibility >= 0.7:
            return ReportFormat.LITERATURE_REVIEW
        else:
            return ReportFormat.GAP_ANALYSIS
    
    if unique_topics >= 3:
        return ReportFormat.COMPARISON
    
    if avg_credibility < 0.6:
        return ReportFormat.FACT_VALIDATION
    
    return ReportFormat.SUMMARY


def extract_comparison_data(findings: List[Finding]) -> Dict[str, List[Finding]]:
    """Extract findings grouped by topic for comparison.
    
    Args:
        findings: List of Finding objects
        
    Returns:
        Dictionary mapping topics to their findings
    """
    comparison_data: Dict[str, List[Finding]] = {}
    
    for finding in findings:
        topic = finding.topic
        if topic not in comparison_data:
            comparison_data[topic] = []
        comparison_data[topic].append(finding)
    
    return comparison_data


def extract_literature_review_structure(
    findings: List[Finding]
) -> Dict[str, Any]:
    """Extract structure for literature review format.
    
    Args:
        findings: List of Finding objects
        
    Returns:
        Dictionary with thematic organization and statistics
    """
    themes = extract_comparison_data(findings)
    
    total_findings = len(findings)
    high_quality_count = sum(1 for f in findings if f.credibility_score >= 0.7)
    
    peer_reviewed = sum(
        1 for f in findings
        if f.citation.source_type and 'peer' in f.citation.source_type.value.lower()
    )
    
    return {
        "themes": themes,
        "statistics": {
            "total_findings": total_findings,
            "high_quality_percentage": (high_quality_count / total_findings * 100) if total_findings > 0 else 0,
            "peer_reviewed_count": peer_reviewed,
            "theme_count": len(themes)
        }
    }


def extract_gap_analysis_structure(
    findings: List[Finding]
) -> Dict[str, Any]:
    """Extract structure for gap analysis format.
    
    Args:
        findings: List of Finding objects
        
    Returns:
        Dictionary with coverage analysis and gap identification
    """
    topics_coverage = extract_comparison_data(findings)
    
    well_covered = []
    under_researched = []
    low_quality = []
    
    for topic, topic_findings in topics_coverage.items():
        count = len(topic_findings)
        avg_credibility = sum(f.credibility_score for f in topic_findings) / count
        
        if count >= 3 and avg_credibility >= 0.7:
            well_covered.append({
                "topic": topic,
                "count": count,
                "avg_credibility": avg_credibility
            })
        elif count < 2:
            under_researched.append({
                "topic": topic,
                "count": count,
                "avg_credibility": avg_credibility
            })
        elif avg_credibility < 0.6:
            low_quality.append({
                "topic": topic,
                "count": count,
                "avg_credibility": avg_credibility
            })
    
    return {
        "coverage": topics_coverage,
        "well_covered_topics": well_covered,
        "under_researched_topics": under_researched,
        "low_quality_topics": low_quality
    }


def get_credibility_warning(score: float) -> Optional[str]:
    """Get warning message for low credibility scores.
    
    Args:
        score: Credibility score (0.0-1.0)
        
    Returns:
        Warning string if score < 0.5, else None
    """
    if score < 0.5:
        return "⚠️ Low Credibility Source"
    return None
