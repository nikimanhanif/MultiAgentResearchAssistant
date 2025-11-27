"""Markdown formatting utilities for report generation."""

from typing import List, Dict, Any, Optional


def format_section(title: str, content: str, level: int = 2) -> str:
    """Format a markdown section with heading and content.
    
    Args:
        title: Section title
        content: Section content
        level: Heading level (1-6)
        
    Returns:
        Formatted markdown section
    """
    if not 1 <= level <= 6:
        raise ValueError("Heading level must be between 1 and 6")
    
    heading_prefix = "#" * level
    return f"{heading_prefix} {title}\n\n{content}\n"


def format_list(items: List[str], ordered: bool = False) -> str:
    """Format a markdown list.
    
    Args:
        items: List items
        ordered: Whether to create ordered list (numbered)
        
    Returns:
        Formatted markdown list
    """
    if not items:
        return ""
    
    if ordered:
        return "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)]) + "\n"
    else:
        return "\n".join([f"- {item}" for item in items]) + "\n"


def format_table(
    headers: List[str],
    rows: List[List[str]],
    alignment: Optional[List[str]] = None
) -> str:
    """Format a markdown table.
    
    Args:
        headers: Column headers
        rows: Table rows (list of lists)
        alignment: Column alignment ('left', 'center', 'right')
        
    Returns:
        Formatted markdown table
    """
    if not headers:
        raise ValueError("Table must have headers")
    if not rows:
        raise ValueError("Table must have at least one row")
    
    num_cols = len(headers)
    for row in rows:
        if len(row) != num_cols:
            raise ValueError(f"All rows must have {num_cols} columns")
    
    if alignment is None:
        alignment = ['left'] * num_cols
    elif len(alignment) != num_cols:
        raise ValueError(f"Alignment must have {num_cols} values")
    
    alignment_symbols = {
        'left': ':--',
        'center': ':-:',
        'right': '--:'
    }
    
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join([
        alignment_symbols.get(align, ':--') for align in alignment
    ]) + " |"
    
    data_rows = "\n".join([
        "| " + " | ".join(row) + " |" for row in rows
    ])
    
    return f"{header_row}\n{separator_row}\n{data_rows}\n"


def format_ranking(
    items: List[Dict[str, Any]],
    rank_key: str = "rank",
    name_key: str = "name",
    score_key: Optional[str] = None
) -> str:
    """Format a ranked list of items.
    
    Args:
        items: List of items with ranking data
        rank_key: Key for rank value
        name_key: Key for item name
        score_key: Optional key for score value
        
    Returns:
        Formatted markdown ranking
    """
    if not items:
        return ""
    
    ranked_items = []
    for item in items:
        rank = item.get(rank_key, "?")
        name = item.get(name_key, "Unknown")
        
        if score_key and score_key in item:
            score = item[score_key]
            ranked_items.append(f"**{rank}. {name}** (Score: {score})")
        else:
            ranked_items.append(f"**{rank}. {name}**")
    
    return "\n".join(ranked_items) + "\n"
