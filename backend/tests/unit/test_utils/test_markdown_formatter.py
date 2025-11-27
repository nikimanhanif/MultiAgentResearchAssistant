"""Unit tests for markdown formatter utilities."""

import pytest
from app.utils.markdown_formatter import (
    format_section,
    format_list,
    format_table,
    format_ranking
)


class TestFormatSection:
    def test_format_section_level_2(self):
        result = format_section("Introduction", "This is content", level=2)
        assert result == "## Introduction\n\nThis is content\n"

    def test_format_section_level_1(self):
        result = format_section("Title", "Content", level=1)
        assert result == "# Title\n\nContent\n"

    def test_format_section_level_6(self):
        result = format_section("Subsection", "Text", level=6)
        assert result == "###### Subsection\n\nText\n"

    def test_format_section_invalid_level_raises_error(self):
        with pytest.raises(ValueError, match="Heading level must be between 1 and 6"):
            format_section("Title", "Content", level=7)

    def test_format_section_zero_level_raises_error(self):
        with pytest.raises(ValueError, match="Heading level must be between 1 and 6"):
            format_section("Title", "Content", level=0)


class TestFormatList:
    def test_format_list_unordered(self):
        items = ["First", "Second", "Third"]
        result = format_list(items, ordered=False)
        assert result == "- First\n- Second\n- Third\n"

    def test_format_list_ordered(self):
        items = ["First", "Second", "Third"]
        result = format_list(items, ordered=True)
        assert result == "1. First\n2. Second\n3. Third\n"

    def test_format_list_empty_returns_empty(self):
        result = format_list([], ordered=False)
        assert result == ""

    def test_format_list_single_item(self):
        result = format_list(["Only item"], ordered=True)
        assert result == "1. Only item\n"


class TestFormatTable:
    def test_format_table_basic(self):
        headers = ["Name", "Age"]
        rows = [["Alice", "30"], ["Bob", "25"]]
        result = format_table(headers, rows)
        expected = "| Name | Age |\n| :-- | :-- |\n| Alice | 30 |\n| Bob | 25 |\n"
        assert result == expected

    def test_format_table_with_alignment(self):
        headers = ["Name", "Score", "Status"]
        rows = [["Alice", "95", "Pass"]]
        result = format_table(headers, rows, alignment=["left", "right", "center"])
        expected = "| Name | Score | Status |\n| :-- | --: | :-: |\n| Alice | 95 | Pass |\n"
        assert result == expected

    def test_format_table_no_headers_raises_error(self):
        with pytest.raises(ValueError, match="Table must have headers"):
            format_table([], [["A", "B"]])

    def test_format_table_no_rows_raises_error(self):
        with pytest.raises(ValueError, match="Table must have at least one row"):
            format_table(["Name", "Age"], [])

    def test_format_table_mismatched_columns_raises_error(self):
        headers = ["A", "B"]
        rows = [["1", "2", "3"]]
        with pytest.raises(ValueError, match="All rows must have 2 columns"):
            format_table(headers, rows)

    def test_format_table_mismatched_alignment_raises_error(self):
        headers = ["A", "B"]
        rows = [["1", "2"]]
        with pytest.raises(ValueError, match="Alignment must have 2 values"):
            format_table(headers, rows, alignment=["left"])


class TestFormatRanking:
    def test_format_ranking_with_score(self):
        items = [
            {"rank": 1, "name": "Item A", "score": 95},
            {"rank": 2, "name": "Item B", "score": 88}
        ]
        result = format_ranking(items, score_key="score")
        expected = "**1. Item A** (Score: 95)\n**2. Item B** (Score: 88)\n"
        assert result == expected

    def test_format_ranking_without_score(self):
        items = [
            {"rank": 1, "name": "First"},
            {"rank": 2, "name": "Second"}
        ]
        result = format_ranking(items)
        expected = "**1. First**\n**2. Second**\n"
        assert result == expected

    def test_format_ranking_empty_returns_empty(self):
        result = format_ranking([])
        assert result == ""

    def test_format_ranking_custom_keys(self):
        items = [
            {"position": 1, "title": "Winner", "points": 100}
        ]
        result = format_ranking(
            items,
            rank_key="position",
            name_key="title",
            score_key="points"
        )
        expected = "**1. Winner** (Score: 100)\n"
        assert result == expected

    def test_format_ranking_missing_keys_uses_defaults(self):
        items = [{"other": "value"}]
        result = format_ranking(items)
        expected = "**?. Unknown**\n"
        assert result == expected
