"""Hybrid judge: analytic verifier when tractable, MLLM judge as fallback.

# PAI-BENCH-2-CHANGE: this is the default for track G in v2. Eliminates the
# v1 single-MLLM bias for the ~60% of items the physics verifier can handle
# (rigid body, contact, fluid heuristic). Intractable scenarios route to the
# `mllm_judge` slot, which is intended to be an EnsembleJudge in production
# so a single MLLM's quirks don't anchor the verdict; a single MLLMJudge
# also satisfies the interface for ad-hoc debugging.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pai_bench.data.schema import GenerationItem
from pai_bench.judge.base import BaseJudge
from pai_bench.judge.physics_judge import PhysicsJudge

logger = logging.getLogger(__name__)


class HybridJudge(BaseJudge):
    judge_type = "hybrid"

    def __init__(
        self,
        physics_judge: PhysicsJudge | None = None,
        mllm_judge: BaseJudge | None = None,
    ):
        self.physics_judge = physics_judge or PhysicsJudge()
        self.mllm_judge = mllm_judge

    def score(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        verdict = self.physics_judge.score(video, item)
        if verdict.get("verifier_type") != "intractable" and verdict.get("score") is not None:
            verdict["judge_type"] = self.physics_judge.judge_type
            return verdict
        if self.mllm_judge is None:
            logger.info(
                "HybridJudge: no MLLM judge configured and item %s is intractable; "
                "returning unscored.",
                item.item_id,
            )
            return {
                "score": None,
                "judge_type": "none",
                "uncertainty": "high",
                "violations": [],
                "verifier_type": "intractable",
            }
        out = self.mllm_judge.score(video, item)
        out.setdefault("judge_type", getattr(self.mllm_judge, "judge_type", "mllm"))
        out.setdefault("uncertainty", "high")
        return out
