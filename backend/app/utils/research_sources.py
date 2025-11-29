"""Curated research sources configuration.

Defines available research sources with metadata for user-facing configuration.
Includes academic sources supported by the Scientific Paper Harvester MCP.

Strategy: "Abstract-First"
Agents should primarily search and read abstracts to determine relevance.
Full-text fetching is restricted to high-value targets only to conserve tokens and time.

"""

from typing import Dict, TypedDict, List


class SourceConfig(TypedDict):
    name: str
    description: str
    categories: List[str]
    mcp_server: str  # Name of the MCP server that handles this source
    priority: int  # 1 (High) to 5 (Low)


CURATED_RESEARCH_SOURCES: Dict[str, SourceConfig] = {
    "arxiv": {
        "name": "arXiv",
        "description": "Preprints in Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering and Systems Science, and Economics.",
        "categories": ["Computer Science", "Physics", "Mathematics", "Engineering"],
        "mcp_server": "scientific-papers",
        "priority": 1
    },
    "openalex": {
        "name": "OpenAlex",
        "description": "Comprehensive index of scholarly papers, authors, institutions, and more. Good for general scientific search.",
        "categories": ["General Science", "Meta-Research"],
        "mcp_server": "scientific-papers",
        "priority": 2
    },
    "pubmed_central": {
        "name": "PubMed Central",
        "description": "Free full-text archive of biomedical and life sciences journal literature at the U.S. National Institutes of Health's National Library of Medicine.",
        "categories": ["Biomedical", "Life Sciences", "Medicine"],
        "mcp_server": "scientific-papers",
        "priority": 1
    },
    "europe_pmc": {
        "name": "Europe PMC",
        "description": "Worldwide life sciences research. Partner to PubMed Central.",
        "categories": ["Biomedical", "Life Sciences"],
        "mcp_server": "scientific-papers",
        "priority": 2
    },
    "biorxiv_medrxiv": {
        "name": "bioRxiv & medRxiv",
        "description": "Preprints for Biology (bioRxiv) and Health Sciences (medRxiv).",
        "categories": ["Biology", "Medicine", "Health Sciences"],
        "mcp_server": "scientific-papers",
        "priority": 2
    },
    "core": {
        "name": "CORE",
        "description": "The world's largest collection of open access research papers.",
        "categories": ["Open Access", "General"],
        "mcp_server": "scientific-papers",
        "priority": 3
    }
}
