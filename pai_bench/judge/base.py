"""Abstract judge interface used by domain scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from pai_bench.data.schema import GenerationItem


class BaseJudge(ABC):
    """Common interface so DomainScorer can swap judges without branching."""

    judge_type: str = "abstract"

    @abstractmethod
    def score(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        """Return {score: float|None, judge_type: str, uncertainty: str, violations: list[str]}."""
