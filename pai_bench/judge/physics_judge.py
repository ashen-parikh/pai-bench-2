"""Analytic physics judge wrapper.

Thin adapter so PhysicsVerifier conforms to the BaseJudge interface, letting
the hybrid judge swap implementations behind a uniform contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pai_bench.data.schema import GenerationItem
from pai_bench.judge.base import BaseJudge
from pai_bench.metrics.physics_verifier import PhysicsVerifier


class PhysicsJudge(BaseJudge):
    judge_type = "analytic"

    def __init__(self, verifier: PhysicsVerifier | None = None):
        self.verifier = verifier or PhysicsVerifier()

    def score(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        result = self.verifier.verify(video, item)
        # Normalise to BaseJudge contract.
        result.setdefault("judge_type", self.judge_type)
        result.setdefault("uncertainty", "low")
        result.setdefault("violations", [])
        return result
