"""
Unit tests for BibTeX export utility.

Tests all pure functions in app.utils.export_bibtex for correctness,
edge cases, and deduplication logic.
"""

import pytest
from app.models.schemas import Citation, Finding, SourceType
from app.utils.export_bibtex import (
    escape_bibtex_chars,
    generate_citation_key,
    source_type_to_bibtex_entry,
    format_bibtex_entry,
    generate_bibtex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_citation(**overrides) -> Citation:
    """Create a Citation with sensible defaults, overridable per-field."""
    defaults = {
        "source": "Test Source",
        "url": "https://example.com/paper",
        "title": "Machine Learning Applications",
        "authors": ["Smith, John", "Doe, Jane"],
        "year": 2023,
        "credibility_score": 0.9,
        "source_type": SourceType.PEER_REVIEWED,
        "doi": "10.1234/test.2023",
        "venue": "Nature Machine Intelligence",
        "is_peer_reviewed": True,
    }
    defaults.update(overrides)
    return Citation(**defaults)


def _make_finding(citation: Citation, **overrides) -> Finding:
    """Create a Finding wrapping the given Citation."""
    defaults = {
        "claim": "Test claim about research",
        "citation": citation,
        "topic": "test-topic",
        "credibility_score": citation.credibility_score or 0.5,
    }
    defaults.update(overrides)
    return Finding(**defaults)


# ===========================================================================
# TestEscapeBibtexChars
# ===========================================================================


class TestEscapeBibtexChars:
    """Tests for escape_bibtex_chars()."""

    def test_escape_bibtex_chars_ampersand_escaped_correctly(self):
        """Ampersand is escaped to \\&."""
        assert escape_bibtex_chars("A & B") == r"A \& B"

    def test_escape_bibtex_chars_percent_escaped_correctly(self):
        """Percent is escaped to \\%."""
        assert escape_bibtex_chars("100%") == r"100\%"

    def test_escape_bibtex_chars_dollar_escaped_correctly(self):
        """Dollar sign is escaped to \\$."""
        assert escape_bibtex_chars("$100") == r"\$100"

    def test_escape_bibtex_chars_hash_escaped_correctly(self):
        """Hash is escaped to \\#."""
        assert escape_bibtex_chars("#1") == r"\#1"

    def test_escape_bibtex_chars_underscore_escaped_correctly(self):
        """Underscore is escaped to \\_."""
        assert escape_bibtex_chars("a_b") == r"a\_b"

    def test_escape_bibtex_chars_multiple_special_chars_all_escaped(self):
        """Multiple special characters are all escaped."""
        result = escape_bibtex_chars("A & B: 100% of $C")
        assert r"\&" in result
        assert r"\%" in result
        assert r"\$" in result

    def test_escape_bibtex_chars_no_special_chars_unchanged(self):
        """Plain text passes through unchanged."""
        text = "Normal text without specials"
        assert escape_bibtex_chars(text) == text

    def test_escape_bibtex_chars_empty_string_returns_empty(self):
        """Empty string returns empty string."""
        assert escape_bibtex_chars("") == ""


# ===========================================================================
# TestGenerateCitationKey
# ===========================================================================


class TestGenerateCitationKey:
    """Tests for generate_citation_key()."""

    def test_generate_citation_key_single_author_correct_format(self):
        """Single author produces lastnameyeartitleword key."""
        citation = _make_citation(
            authors=["Smith, John"],
            year=2023,
            title="Machine Learning in Healthcare",
        )
        key = generate_citation_key(citation, 0, set())
        assert key == "smith2023machine"

    def test_generate_citation_key_multiple_authors_uses_first(self):
        """Multiple authors: only the first author's last name is used."""
        citation = _make_citation(
            authors=["Garcia, Maria", "Lee, Wei"],
            year=2024,
            title="Deep Learning Survey",
        )
        key = generate_citation_key(citation, 0, set())
        assert key.startswith("garcia2024")

    def test_generate_citation_key_no_authors_uses_fallback(self):
        """No authors falls back to source{index}_{year} format."""
        citation = _make_citation(authors=None, year=2023)
        key = generate_citation_key(citation, 5, set())
        assert key == "source5_2023"

    def test_generate_citation_key_empty_authors_uses_fallback(self):
        """Empty authors list falls back to source{index}_{year}."""
        citation = _make_citation(authors=[], year=2023)
        key = generate_citation_key(citation, 3, set())
        assert key == "source3_2023"

    def test_generate_citation_key_no_title_uses_author_year_only(self):
        """No title produces key from author + year with no title part."""
        citation = _make_citation(
            authors=["Jones, Alice"],
            year=2022,
            title=None,
        )
        key = generate_citation_key(citation, 0, set())
        assert key == "jones2022"

    def test_generate_citation_key_no_year_omits_year(self):
        """No year omits the year portion from the key."""
        citation = _make_citation(
            authors=["Brown, Bob"],
            year=None,
            title="Some Paper",
        )
        key = generate_citation_key(citation, 0, set())
        assert key == "brownsome"

    def test_generate_citation_key_no_year_no_author_fallback(self):
        """No year and no author produces source{index} only."""
        citation = _make_citation(authors=None, year=None)
        key = generate_citation_key(citation, 7, set())
        assert key == "source7"

    def test_generate_citation_key_unicode_author_ascii_safe(self):
        """Unicode author names are converted to ASCII-safe keys."""
        citation = _make_citation(
            authors=["Müller, Hans"],
            year=2023,
            title="Quantum Computing",
        )
        key = generate_citation_key(citation, 0, set())
        assert key == "muller2023quantum"

    def test_generate_citation_key_duplicate_key_appends_suffix(self):
        """Duplicate keys get a/b/c suffixes."""
        citation = _make_citation(
            authors=["Smith, John"],
            year=2023,
            title="Machine Learning",
        )
        existing = {"smith2023machine"}
        key = generate_citation_key(citation, 0, existing)
        assert key == "smith2023machinea"

    def test_generate_citation_key_multiple_duplicates_increment_suffix(self):
        """Multiple duplicates increment through alphabet."""
        citation = _make_citation(
            authors=["Smith, John"],
            year=2023,
            title="Machine Learning",
        )
        existing = {"smith2023machine", "smith2023machinea"}
        key = generate_citation_key(citation, 0, existing)
        assert key == "smith2023machineb"

    def test_generate_citation_key_title_skips_common_words(self):
        """Common words (the, a, an, of, etc.) are skipped in title."""
        citation = _make_citation(
            authors=["Doe, Jane"],
            year=2021,
            title="The Impact of AI on Healthcare",
        )
        key = generate_citation_key(citation, 0, set())
        assert key == "doe2021impact"

    def test_generate_citation_key_author_first_last_format(self):
        """Author in 'First Last' format extracts last name correctly."""
        citation = _make_citation(
            authors=["John Smith"],
            year=2023,
            title="Neural Networks",
        )
        key = generate_citation_key(citation, 0, set())
        assert key == "smith2023neural"


# ===========================================================================
# TestSourceTypeToBibtexEntry
# ===========================================================================


class TestSourceTypeToBibtexEntry:
    """Tests for source_type_to_bibtex_entry()."""

    def test_peer_reviewed_maps_to_article(self):
        assert source_type_to_bibtex_entry(SourceType.PEER_REVIEWED) == "article"

    def test_conference_maps_to_inproceedings(self):
        assert source_type_to_bibtex_entry(SourceType.CONFERENCE) == "inproceedings"

    def test_book_maps_to_book(self):
        assert source_type_to_bibtex_entry(SourceType.BOOK) == "book"

    def test_thesis_maps_to_phdthesis(self):
        assert source_type_to_bibtex_entry(SourceType.THESIS) == "phdthesis"

    def test_preprint_maps_to_misc(self):
        assert source_type_to_bibtex_entry(SourceType.PREPRINT) == "misc"

    def test_news_maps_to_misc(self):
        assert source_type_to_bibtex_entry(SourceType.NEWS) == "misc"

    def test_blog_maps_to_misc(self):
        assert source_type_to_bibtex_entry(SourceType.BLOG) == "misc"

    def test_website_maps_to_misc(self):
        assert source_type_to_bibtex_entry(SourceType.WEBSITE) == "misc"

    def test_none_maps_to_misc(self):
        assert source_type_to_bibtex_entry(None) == "misc"


# ===========================================================================
# TestFormatBibtexEntry
# ===========================================================================


class TestFormatBibtexEntry:
    """Tests for format_bibtex_entry()."""

    def test_format_bibtex_entry_article_with_all_fields(self):
        """Article entry includes journal, doi, url, author, title, year."""
        citation = _make_citation()
        result = format_bibtex_entry("smith2023machine", citation, "article")

        assert result.startswith("@article{smith2023machine,")
        assert "author = {Smith, John and Doe, Jane}" in result
        assert "title = {{Machine Learning Applications}}" in result
        assert "year = {2023}" in result
        assert "journal = {Nature Machine Intelligence}" in result
        assert "doi = {10.1234/test.2023}" in result
        assert "url = {https://example.com/paper}" in result
        assert "Credibility: 0.90" in result
        assert "Peer-reviewed" in result
        assert result.endswith("}")

    def test_format_bibtex_entry_misc_with_minimal_fields(self):
        """Misc entry with only source name works correctly."""
        citation = _make_citation(
            authors=None,
            title=None,
            year=None,
            doi=None,
            url=None,
            venue=None,
            credibility_score=None,
            is_peer_reviewed=None,
        )
        result = format_bibtex_entry("source0", citation, "misc")

        assert result.startswith("@misc{source0,")
        assert "author = {Unknown}" in result
        assert "title = {{Test Source}}" in result  # Falls back to source name
        assert "year" not in result.split("author")[1].split("title")[0]

    def test_format_bibtex_entry_inproceedings_uses_booktitle(self):
        """Inproceedings entry uses booktitle instead of journal."""
        citation = _make_citation(
            source_type=SourceType.CONFERENCE,
            venue="NeurIPS 2023",
        )
        result = format_bibtex_entry("key1", citation, "inproceedings")

        assert "@inproceedings{key1," in result
        assert "booktitle = {NeurIPS 2023}" in result
        assert "journal" not in result

    def test_format_bibtex_entry_special_chars_in_title_escaped(self):
        """Special characters in title are escaped."""
        citation = _make_citation(title="Cost $100 & Performance: 95%")
        result = format_bibtex_entry("key1", citation, "article")

        assert r"\$" in result
        assert r"\&" in result
        assert r"\%" in result

    def test_format_bibtex_entry_not_peer_reviewed_note(self):
        """Not peer-reviewed flag appears in note."""
        citation = _make_citation(is_peer_reviewed=False, credibility_score=0.3)
        result = format_bibtex_entry("key1", citation, "misc")

        assert "Not peer-reviewed" in result
        assert "Credibility: 0.30" in result


# ===========================================================================
# TestGenerateBibtex
# ===========================================================================


class TestGenerateBibtex:
    """Tests for generate_bibtex()."""

    def test_generate_bibtex_single_finding_produces_valid_output(self):
        """Single finding produces a header and one entry."""
        citation = _make_citation()
        findings = [_make_finding(citation)]
        result = generate_bibtex(findings)

        assert result.startswith("% Bibliography exported from MultiAgent Research Assistant")
        assert "@article{" in result
        assert result.count("@") == 1

    def test_generate_bibtex_multiple_findings_same_source_deduplicates(self):
        """Multiple findings with same DOI produce only one entry."""
        citation = _make_citation(doi="10.1234/same")
        findings = [
            _make_finding(citation, claim="Claim 1"),
            _make_finding(citation, claim="Claim 2"),
            _make_finding(citation, claim="Claim 3"),
        ]
        result = generate_bibtex(findings)

        assert result.count("@article{") == 1

    def test_generate_bibtex_multiple_findings_different_sources(self):
        """Findings with different DOIs produce multiple entries."""
        c1 = _make_citation(doi="10.1234/first", title="First Paper")
        c2 = _make_citation(doi="10.1234/second", title="Second Paper")
        findings = [_make_finding(c1), _make_finding(c2)]
        result = generate_bibtex(findings)

        assert result.count("@article{") == 2

    def test_generate_bibtex_empty_findings_returns_header_only(self):
        """Empty findings list returns just the header comment."""
        result = generate_bibtex([])

        assert "% Bibliography exported from MultiAgent Research Assistant" in result
        assert "@" not in result

    def test_generate_bibtex_findings_with_doi_deduplicates_by_doi(self):
        """Citations with same DOI but different URLs are deduplicated."""
        c1 = _make_citation(doi="10.1234/paper", url="https://a.com")
        c2 = _make_citation(doi="10.1234/paper", url="https://b.com")
        findings = [_make_finding(c1), _make_finding(c2)]
        result = generate_bibtex(findings)

        assert result.count("@") == 1

    def test_generate_bibtex_findings_deduplicates_by_url_when_no_doi(self):
        """Citations with no DOI but same URL are deduplicated."""
        c1 = _make_citation(doi=None, url="https://same.com/paper")
        c2 = _make_citation(doi=None, url="https://same.com/paper")
        findings = [_make_finding(c1), _make_finding(c2)]
        result = generate_bibtex(findings)

        assert result.count("@") == 1

    def test_generate_bibtex_findings_deduplicates_by_title(self):
        """Citations with no DOI/URL but same title are deduplicated."""
        c1 = _make_citation(doi=None, url=None, title="Same Title")
        c2 = _make_citation(doi=None, url=None, title="Same Title")
        findings = [_make_finding(c1), _make_finding(c2)]
        result = generate_bibtex(findings)

        assert result.count("@") == 1

    def test_generate_bibtex_mixed_source_types(self):
        """Findings with different source types produce correct entry types."""
        c1 = _make_citation(
            source_type=SourceType.PEER_REVIEWED,
            doi="10.1/a",
            title="Paper A",
        )
        c2 = _make_citation(
            source_type=SourceType.CONFERENCE,
            doi="10.1/b",
            title="Paper B",
        )
        c3 = _make_citation(
            source_type=SourceType.WEBSITE,
            doi="10.1/c",
            title="Paper C",
        )
        findings = [_make_finding(c1), _make_finding(c2), _make_finding(c3)]
        result = generate_bibtex(findings)

        assert "@article{" in result
        assert "@inproceedings{" in result
        assert "@misc{" in result
