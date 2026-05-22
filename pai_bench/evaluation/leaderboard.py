"""Leaderboard aggregation and PAI-Index computation.

# PAI-BENCH-2-CHANGE: PAI-Index weights U (accuracy + multihop) and CF higher
# than G/C. Understanding + causal reasoning correlate more strongly with the
# DV track than raw generation fidelity in v1 pilot data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from pai_bench.data.schema import TrackScore

logger = logging.getLogger(__name__)

# Weights mirror config/track_config.yaml::pai_index_weights.
DEFAULT_WEIGHTS = {
    "G_domain": 0.20,
    "C_fidelity": 0.15,
    "U_accuracy": 0.30,
    "U_multihop": 0.20,
    "CF_delta_accuracy": 0.15,
}


def pai_index(scores: dict[str, TrackScore], weights: dict[str, float] | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    g = scores.get("G")
    c = scores.get("C")
    u = scores.get("U")
    cf = scores.get("CF")
    parts = []
    if g:
        parts.append(w["G_domain"] * float(g.scores.get("domain_score", 0.0)))
    if c:
        # Mean of fidelity-style metrics for a single composite.
        fidelity = [c.scores.get(k, 0.0) for k in ("blur_ssim", "edge_f1", "depth_si_rmse", "mask_miou")]
        parts.append(w["C_fidelity"] * (sum(fidelity) / len(fidelity)))
    if u:
        parts.append(w["U_accuracy"] * float(u.scores.get("accuracy", 0.0)))
        parts.append(w["U_multihop"] * float(u.scores.get("multihop_chain_accuracy", 0.0)))
    if cf:
        parts.append(w["CF_delta_accuracy"] * float(cf.scores.get("counterfactual_delta_accuracy", 0.0)))
    return float(sum(parts))


class Leaderboard:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._rows: dict[str, dict[str, TrackScore]] = {}

    def add_model(self, model_id: str, scores: dict[str, TrackScore]) -> None:
        self._rows[model_id] = scores

    def rank(self) -> pd.DataFrame:
        rows = []
        for model_id, tracks in self._rows.items():
            row = {"model_id": model_id, "pai_index": pai_index(tracks, self.weights)}
            for tid, ts in tracks.items():
                for k, v in ts.scores.items():
                    row[f"{tid}.{k}"] = v
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("pai_index", ascending=False).reset_index(drop=True)
        return df

    def spearman_validity(self, dv_scores: dict[str, float]) -> dict[str, float]:
        """Spearman rho between each track metric and held-out DV success.

        dv_scores: model_id -> single composite task success rate.
        """
        if len(dv_scores) < 3:
            return {}
        model_ids = [m for m in self._rows if m in dv_scores]
        out: dict[str, float] = {}
        # Build per-metric correlation against dv_scores.
        df = self.rank()
        for col in df.columns:
            if col in ("model_id", "pai_index"):
                continue
            xs = []
            ys = []
            for m in model_ids:
                row = df[df.model_id == m]
                if row.empty:
                    continue
                xs.append(float(row[col].iloc[0]))
                ys.append(float(dv_scores[m]))
            if len(xs) >= 3:
                rho, _ = spearmanr(xs, ys)
                if rho == rho:
                    out[col] = float(rho)
        return out

    def to_markdown(self, path: Path) -> None:
        df = self.rank()
        Path(path).write_text("# PAI-Bench 2 leaderboard\n\n" + df.to_markdown(index=False))
