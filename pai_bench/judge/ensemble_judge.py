"""MLLM ensemble judge.

# PAI-BENCH-2-CHANGE: replaces v1's single Qwen3-VL judge. Runs N independent
# MLLM judges and reports:
#   - per-judge raw verdicts
#   - aggregated score (median by default; resistant to a single outlier judge)
#   - inter-judge std and range
#   - disagreement_flagged bool when std exceeds the configured threshold
#
# The ensemble fits the BaseJudge interface so it slots into HybridJudge in
# place of the single MLLM fallback. When fewer than 2 judges are available,
# it degrades gracefully and flags uncertainty='high'.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from pai_bench.data.schema import GenerationItem
from pai_bench.judge.base import BaseJudge
from pai_bench.judge.mllm_judge import MLLMJudge

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ("gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-latest")


class EnsembleJudge(BaseJudge):
    judge_type = "ensemble"

    def __init__(
        self,
        judges: list[BaseJudge] | None = None,
        aggregation: Literal["median", "mean", "majority"] = "median",
        disagreement_threshold: float = 0.25,
    ):
        if not judges:
            raise ValueError("EnsembleJudge requires at least one inner judge")
        self.judges = judges
        self.aggregation = aggregation
        self.disagreement_threshold = disagreement_threshold

    @classmethod
    def from_model_names(
        cls,
        model_names: tuple[str, ...] = DEFAULT_MODELS,
        **kwargs,
    ) -> "EnsembleJudge":
        judges = [MLLMJudge(model=name) for name in model_names]
        return cls(judges=judges, **kwargs)

    def score(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        per_judge: list[dict[str, Any]] = []
        for j in self.judges:
            try:
                verdict = j.score(video, item)
            except Exception as exc:                          # individual judge failure
                logger.warning("ensemble member %s failed for %s: %s",
                               getattr(j, "judge_type", j.__class__.__name__),
                               item.item_id, exc)
                continue
            if verdict.get("score") is not None:
                per_judge.append(verdict)

        if not per_judge:
            return {
                "score": None,
                "judge_type": self.judge_type,
                "uncertainty": "high",
                "violations": [],
                "per_judge": [],
                "inter_judge_std": None,
                "inter_judge_range": None,
                "disagreement_flagged": False,
                "n_judges": 0,
            }

        scores = [float(v["score"]) for v in per_judge]
        agg = self._aggregate(scores)

        if len(scores) >= 2:
            std = float(np.std(scores, ddof=0))
            rng = float(max(scores) - min(scores))
        else:
            std = 0.0
            rng = 0.0
        flagged = std >= self.disagreement_threshold

        # Union of violations so callers can see the worst-case complaints.
        violations: list[str] = []
        for v in per_judge:
            violations.extend(v.get("violations", []))
        violations = sorted(set(violations))

        return {
            "score": float(agg),
            "judge_type": self.judge_type,
            "uncertainty": "high" if flagged or len(scores) < 2 else "medium",
            "violations": violations,
            "per_judge": per_judge,
            "per_judge_scores": scores,
            "inter_judge_std": std,
            "inter_judge_range": rng,
            "disagreement_flagged": bool(flagged),
            "n_judges": len(per_judge),
            "aggregation": self.aggregation,
        }

    def _aggregate(self, scores: list[float]) -> float:
        if self.aggregation == "median":
            return float(np.median(scores))
        if self.aggregation == "mean":
            return float(np.mean(scores))
        if self.aggregation == "majority":
            votes = [int(s >= 0.5) for s in scores]
            return float(sum(votes) / len(votes))
        raise ValueError(f"unknown aggregation: {self.aggregation!r}")
