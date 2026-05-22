"""Track G: unconditional video generation scoring.

Quality Score + Domain Score, weighted per config/track_config.yaml.
# PAI-BENCH-2-CHANGE: domain_score_weight raised to 0.7; hybrid_judge default.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

from pai_bench.data.schema import (
    BenchmarkItem, GenerationItem, ModelPrediction, TrackScore,
)
from pai_bench.judge.hybrid_judge import HybridJudge
from pai_bench.metrics import quality
from pai_bench.metrics.domain import DomainScorer
from pai_bench.tracks.base import BaseTrack

logger = logging.getLogger(__name__)


def _read_video(path: str | Path) -> np.ndarray:
    arr = iio.imread(path, plugin="pyav")
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr.astype(np.float32) / 255.0


class GenerationTrack(BaseTrack):
    track_id = "G"
    quality_weight = 0.3
    domain_weight = 0.7

    def __init__(self, config_dir: Path | None = None, judge: HybridJudge | None = None):
        super().__init__(config_dir)
        self.judge = judge or HybridJudge()
        self.domain_scorer = DomainScorer(mllm_judge=self.judge.mllm_judge)

    def load_items(self, data_dir: Path) -> list[BenchmarkItem]:
        items_dir = Path(data_dir) / "items" / "G"
        out: list[BenchmarkItem] = []
        for p in sorted(items_dir.glob("*.json")):
            payload = json.loads(p.read_text())
            gi = GenerationItem(**payload)
            out.append(BenchmarkItem(track="G", item=gi))
        return out

    def evaluate(
        self,
        model_fn: Callable,
        items: list[BenchmarkItem],
    ) -> list[ModelPrediction]:
        preds: list[ModelPrediction] = []
        for bi in tqdm(items, desc="G eval"):
            gi: GenerationItem = bi.item              # type: ignore[assignment]
            response = model_fn({"prompt": gi.prompt, "item_id": gi.item_id})
            preds.append(ModelPrediction(
                model_id=response.get("model_id", "unknown"),
                item_id=gi.item_id,
                track=self.track_id,
                prediction={"video_path": response["video_path"]},
                latency_ms=response.get("latency_ms"),
            ))
        return preds

    def score(
        self,
        predictions: list[ModelPrediction],
        items: list[BenchmarkItem],
    ) -> TrackScore:
        items_by_id = {bi.item.item_id: bi.item for bi in items}        # type: ignore[union-attr]
        per_domain: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        per_physics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        quality_scores: list[float] = []
        domain_scores: list[float] = []
        intractable_count = 0

        for p in tqdm(predictions, desc="G score"):
            gi = items_by_id.get(p.item_id)
            if gi is None:
                continue
            video_path = p.prediction["video_path"] if isinstance(p.prediction, dict) else p.prediction
            try:
                video = _read_video(video_path)
            except Exception as exc:
                logger.warning("failed to read %s: %s", video_path, exc)
                continue

            q = {
                "subject_consistency": quality.subject_consistency(video),
                "background_consistency": quality.background_consistency(video),
                "motion_smoothness": quality.motion_smoothness(video),
                "aesthetic_quality": quality.aesthetic_quality(video),
                "imaging_quality": quality.imaging_quality(video),
                "overall_consistency": quality.overall_consistency(video, gi.prompt),
            }
            qmean = float(np.mean(list(q.values())))
            quality_scores.append(qmean)

            verdict = self.judge.score(video, gi)
            dscore = verdict.get("score")
            if dscore is None:
                intractable_count += 1
            else:
                domain_scores.append(float(dscore))

            dom = gi.domain if isinstance(gi.domain, str) else gi.domain.value
            phys = gi.physics_category if isinstance(gi.physics_category, str) else gi.physics_category.value
            per_domain[dom]["quality"].append(qmean)
            if dscore is not None:
                per_domain[dom]["domain"].append(float(dscore))
                per_physics[phys]["domain"].append(float(dscore))
            per_physics[phys]["quality"].append(qmean)

        qmean = float(np.mean(quality_scores)) if quality_scores else 0.0
        dmean = float(np.mean(domain_scores)) if domain_scores else 0.0
        overall = self.quality_weight * qmean + self.domain_weight * dmean

        agg = lambda d: {k: float(np.mean(v)) if v else 0.0 for k, v in d.items()}
        return TrackScore(
            track=self.track_id,
            model_id=(predictions[0].model_id if predictions else "unknown"),
            n_items=len(items),
            scores={
                "quality_score": qmean,
                "domain_score": dmean,
                "overall_g_score": overall,
                "intractable_items": float(intractable_count),
            },
            per_domain={d: agg(v) for d, v in per_domain.items()},
            per_physics_category={p: agg(v) for p, v in per_physics.items()},
        )
