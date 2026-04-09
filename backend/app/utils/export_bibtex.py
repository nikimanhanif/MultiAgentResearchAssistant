"""
BibTeX export utility.

Pure functions for converting Citation/Finding objects into BibTeX format.
No FastAPI or persistence dependencies — fully testable in isolation.
"""

import re
import unicodedata
from typing import Optional, Set, List

from app.models.schemas import Citation, Finding, SourceType


# LaTeX special characters that must be escaped in BibTeX fields
_BIBTEX_SPECIAL_CHARS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Common words to skip when extracting a title word for citation keys
_SKIP_WORDS = {"the", "a", "an", "of", "in", "on", "for", "and", "to", "with", "is"}

# Mapping from SourceType to BibTeX entry type
_SOURCE_TYPE_MAP = {
    SourceType.PEER_REVIEWED: "article",
    SourceType.CONFERENCE: "inproceedings",
    SourceType.BOOK: "book",
    SourceType.THESIS: "phdthesis",
}


def escape_bibtex_chars(text: str) -> str:
    """
    Escape LaTeX special characters for safe inclusion in BibTeX fields.

    Args:
        text: Raw text string.

    Returns:
        String with special characters escaped.
    """
    if not text:
        return text

    result = []
    for char in text:
        result.append(_BIBTEX_SPECIAL_CHARS.get(char, char))
    return "".join(result)


def _to_ascii_key(text: str) -> str:
    """Normalize text to ASCII-safe lowercase for use in citation keys."""
    # Decompose unicode characters and strip combining marks
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    # Keep only alphanumeric characters
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


def generate_citation_key(
    citation: Citation,
    index: int,
    existing_keys: Set[str],
) -> str:
    """
    Generate a unique BibTeX citation key from citation metadata.

    Produces keys like ``smith2023machine`` from first author + year + first
    significant title word.  Falls back to ``source{index}_{year}`` when
    metadata is sparse.  Appends ``a``, ``b``, ``c`` … to deduplicate.

    Args:
        citation: The citation to generate a key for.
        index: Positional index (used in fallback key).
        existing_keys: Set of already-used keys for deduplication.

    Returns:
        A unique citation key string.
    """
    # Extract author last name
    author_part = ""
    if citation.authors and len(citation.authors) > 0:
        first_author = citation.authors[0]
        # Handle "Last, First" and "First Last" formats
        if "," in first_author:
            last_name = first_author.split(",")[0].strip()
        else:
            parts = first_author.strip().split()
            last_name = parts[-1] if parts else ""
        author_part = _to_ascii_key(last_name)

    # Year part
    year_part = str(citation.year) if citation.year else ""

    # Title word — first significant word
    title_part = ""
    if citation.title:
        words = citation.title.split()
        for word in words:
            cleaned = _to_ascii_key(word)
            if cleaned and cleaned not in _SKIP_WORDS:
                title_part = cleaned
                break

    # Build key or use fallback
    if author_part:
        base_key = f"{author_part}{year_part}{title_part}"
    else:
        base_key = f"source{index}_{year_part}" if year_part else f"source{index}"

    # Deduplicate
    key = base_key
    if key in existing_keys:
        for suffix in "abcdefghijklmnopqrstuvwxyz":
            candidate = f"{base_key}{suffix}"
            if candidate not in existing_keys:
                key = candidate
                break

    existing_keys.add(key)
    return key


def source_type_to_bibtex_entry(source_type: Optional[SourceType]) -> str:
    """
    Map a SourceType enum value to a BibTeX entry type string.

    Args:
        source_type: The source type classification, or None.

    Returns:
        BibTeX entry type (e.g. ``"article"``, ``"inproceedings"``, ``"misc"``).
    """
    if source_type is None:
        return "misc"
    return _SOURCE_TYPE_MAP.get(source_type, "misc")


def format_bibtex_entry(key: str, citation: Citation, entry_type: str) -> str:
    """
    Format a single BibTeX entry string from a Citation.

    Args:
        key: The citation key.
        citation: Citation data.
        entry_type: BibTeX entry type (article, inproceedings, etc.).

    Returns:
        A complete BibTeX entry string.
    """
    lines = [f"@{entry_type}{{{key},"]

    # Author
    if citation.authors and len(citation.authors) > 0:
        authors_str = " and ".join(citation.authors)
        lines.append(f"  author = {{{escape_bibtex_chars(authors_str)}}},")
    else:
        lines.append("  author = {Unknown},")

    # Title (double-braced to preserve capitalization)
    title = citation.title or citation.source or "Untitled"
    lines.append(f"  title = {{{{{escape_bibtex_chars(title)}}}}},")

    # Year
    if citation.year:
        lines.append(f"  year = {{{citation.year}}},")

    # Venue — field name depends on entry type
    if citation.venue:
        escaped_venue = escape_bibtex_chars(citation.venue)
        if entry_type == "article":
            lines.append(f"  journal = {{{escaped_venue}}},")
        elif entry_type == "inproceedings":
            lines.append(f"  booktitle = {{{escaped_venue}}},")
        else:
            lines.append(f"  howpublished = {{{escaped_venue}}},")

    # DOI
    if citation.doi:
        lines.append(f"  doi = {{{citation.doi}}},")

    # URL
    if citation.url:
        lines.append(f"  url = {{{citation.url}}},")

    # Note with credibility info
    note_parts = []
    if citation.credibility_score is not None:
        note_parts.append(f"Credibility: {citation.credibility_score:.2f}")
    if citation.is_peer_reviewed is not None:
        note_parts.append(
            "Peer-reviewed" if citation.is_peer_reviewed else "Not peer-reviewed"
        )
    if note_parts:
        lines.append(f"  note = {{{escape_bibtex_chars(', '.join(note_parts))}}},")

    lines.append("}")
    return "\n".join(lines)


def generate_bibtex(findings: List[Finding]) -> str:
    """
    Generate a complete BibTeX file from a list of research findings.

    Deduplicates citations by DOI, then URL, then title before formatting.

    Args:
        findings: List of Finding objects containing citations.

    Returns:
        Full ``.bib`` file content as a string.
    """
    header = "% Bibliography exported from MultiAgent Research Assistant\n"

    if not findings:
        return header

    # Deduplicate citations: prefer DOI > URL > title as identity key
    seen: Set[str] = set()
    unique_citations: List[Citation] = []

    for finding in findings:
        citation = finding.citation
        identity = None

        if citation.doi:
            identity = f"doi:{citation.doi}"
        elif citation.url:
            identity = f"url:{citation.url}"
        elif citation.title:
            identity = f"title:{citation.title.lower().strip()}"
        else:
            identity = f"source:{citation.source.lower().strip()}"

        if identity not in seen:
            seen.add(identity)
            unique_citations.append(citation)

    # Generate entries
    existing_keys: Set[str] = set()
    entries: List[str] = []

    for idx, citation in enumerate(unique_citations):
        entry_type = source_type_to_bibtex_entry(citation.source_type)
        key = generate_citation_key(citation, idx, existing_keys)
        entry = format_bibtex_entry(key, citation, entry_type)
        entries.append(entry)

    return header + "\n" + "\n\n".join(entries) + "\n"
