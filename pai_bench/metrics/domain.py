"""Domain Score: physics plausibility judge that prefers analytic verifier output
and falls back to MLLM judge for intractable scenarios with an uncertainty flag.

# PAI-BENCH-2-CHANGE: hybrid routing replaces v1's MLLM-as-Judge default.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pai_bench.data.schema import GenerationItem
from pai_bench.metrics.physics_verifier import PhysicsVerifier

logger = logging.getLogger(__name__)


class DomainScorer:
    """Decides whether physics_verifier or MLLM judge handles a given item."""

    def __init__(self, verifier: PhysicsVerifier | None = None, mllm_judge=None):
        self.verifier = verifier or PhysicsVerifier()
        self.mllm_judge = mllm_judge

    def score(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        verification = self.verifier.verify(video, item)
        if verification.get("verifier_type") == "intractable":
            if self.mllm_judge is None:
                logger.warning(
                    "Domain score for item %s is intractable analytically and no "
                    "MLLM judge configured; returning None.",
                    item.item_id,
                )
                return {
                    "score": None,
                    "judge_type": "none",
                    "uncertainty": "high",
                    "violations": [],
                }
            result = self.mllm_judge.score(video, item)
            result.setdefault("violations", [])
            result["judge_type"] = "mllm"
            result["uncertainty"] = "high"
            return result
        verification["judge_type"] = "analytic"
        verification.setdefault("uncertainty", "low")
        return verification
