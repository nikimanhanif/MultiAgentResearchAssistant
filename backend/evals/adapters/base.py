"""
Abstract base adapter for benchmark datasets.

All dataset adapters must implement this interface so the runner
can work with any benchmark source uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from evals.models import BenchmarkCase


class BenchmarkAdapter(ABC):
    """Converts a dataset source into a list of BenchmarkCase objects."""

    @abstractmethod
    def load_cases(self, source: str) -> List[BenchmarkCase]:
        """
        Load benchmark cases from a source path or identifier.

        Args:
            source: File path, URL, or dataset identifier.

        Returns:
            List of BenchmarkCase objects ready for the runner.
        """
        ...

    def adapt_case(self, raw: Dict[str, Any]) -> BenchmarkCase:
        """
        Convert a single raw record into a BenchmarkCase.

        Subclasses should override this to handle dataset-specific
        field mappings. The default implementation assumes the raw
        dict already matches BenchmarkCase fields.
        """
        return BenchmarkCase(**raw)
