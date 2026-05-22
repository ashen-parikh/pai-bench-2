"""Track DV: downstream validity.

# PAI-BENCH-2-CHANGE: new maintainer-held track. Closes the Goodhart loop by
# checking whether benchmark scores predict performance on held-out real tasks
# (robot grasping, AV trajectory prediction, physics-sim fidelity).
#
# This track is not run client-side; the runner skips it and the leaderboard
# imports DV scores from a maintainer-controlled artifact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from scipy.stats import spearmanr

from pai_bench.data.schema import BenchmarkItem, ModelPrediction, TrackScore
from pai_bench.tracks.base import BaseTrack

logger = logging.getLogger(__name__)

DV_TASKS = ("robot_grasping", "av_trajectory_prediction", "physics_sim_fidelity")


class DownstreamTrack(BaseTrack):
    track_id = "DV"

    def load_items(self, data_dir: Path) -> list[BenchmarkItem]:
        logger.info("DV is maintainer-held and not run client-side; returning [].")
        return []

    def evaluate(
        self,
        model_fn: Callable,
        items: list[BenchmarkItem],
    ) -> list[ModelPrediction]:
        # Held out; model_fn is never called client-side.
        return []

    def score(
        self,
        predictions: list[ModelPrediction],
        items: list[BenchmarkItem],
    ) -> TrackScore:
        return TrackScore(
            track=self.track_id,
            model_id="held_out",
            n_items=0,
            scores={},
            per_domain={},
            per_physics_category={},
        )

    @staticmethod
    def compute_validity(
        model_track_scores: dict[str, dict[str, float]],
        task_success: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Compute Spearman rho between each track metric and held-out task success.

        Args:
            model_track_scores: model_id -> {track_id: aggregate_score}.
            task_success: model_id -> {task_name: success_rate}.

        Returns:
            {track_id: rho, "overall_predictive_validity": mean_rho}.
        """
        if not model_track_scores or not task_success:
            return {}
        model_ids = [m for m in model_track_scores if m in task_success]
        if len(model_ids) < 3:
            logger.warning("Need >=3 models for Spearman; got %d", len(model_ids))
            return {}
        out: dict[str, float] = {}
        track_ids = {t for v in model_track_scores.values() for t in v}
        for t in sorted(track_ids):
            xs = [model_track_scores[m].get(t, 0.0) for m in model_ids]
            # Mean across DV subtasks; falls back to single task if only one.
            ys = [
                sum(task_success[m].get(task, 0.0) for task in DV_TASKS) / len(DV_TASKS)
                for m in model_ids
            ]
            rho, _ = spearmanr(xs, ys)
            out[f"spearman_rho_vs_{t}"] = float(rho) if rho == rho else 0.0
        rhos = [v for v in out.values() if v == v]
        out["overall_predictive_validity"] = float(sum(rhos) / len(rhos)) if rhos else 0.0
        return out
