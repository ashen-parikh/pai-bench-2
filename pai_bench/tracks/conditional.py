"""Track C: conditional video generation scoring.

# PAI-BENCH-2-CHANGE: robustness_score is computed across degradation levels
# 0..3 of the control signal, capturing graceful falloff rather than a single
# fidelity point.
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
    BenchmarkItem, ConditionalItem, ModelPrediction, TrackScore,
)
from pai_bench.metrics import control, diversity
from pai_bench.tracks.base import BaseTrack

logger = logging.getLogger(__name__)


def _read_video(path: str | Path) -> np.ndarray:
    arr = iio.imread(path, plugin="pyav")
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr.astype(np.float32) / 255.0


class ConditionalTrack(BaseTrack):
    track_id = "C"

    def load_items(self, data_dir: Path) -> list[BenchmarkItem]:
        items_dir = Path(data_dir) / "items" / "C"
        out: list[BenchmarkItem] = []
        for p in sorted(items_dir.glob("*.json")):
            payload = json.loads(p.read_text())
            ci = ConditionalItem(**payload)
            out.append(BenchmarkItem(track="C", item=ci))
        return out

    def evaluate(
        self,
        model_fn: Callable,
        items: list[BenchmarkItem],
    ) -> list[ModelPrediction]:
        preds: list[ModelPrediction] = []
        for bi in tqdm(items, desc="C eval"):
            ci: ConditionalItem = bi.item              # type: ignore[assignment]
            response = model_fn({
                "prompt": ci.prompt,
                "item_id": ci.item_id,
                "control_signals": ci.control_signals,
                "variant_prompts": ci.variant_prompts,
            })
            # response: {model_id, video_paths_by_level: {0: path, 1: path, ...},
            #            variant_video_paths: [path, ...]}
            preds.append(ModelPrediction(
                model_id=response.get("model_id", "unknown"),
                item_id=ci.item_id,
                track=self.track_id,
                prediction={
                    "video_paths_by_level": response["video_paths_by_level"],
                    "variant_video_paths": response.get("variant_video_paths", []),
                },
                latency_ms=response.get("latency_ms"),
            ))
        return preds

    def score(
        self,
        predictions: list[ModelPrediction],
        items: list[BenchmarkItem],
    ) -> TrackScore:
        items_by_id = {bi.item.item_id: bi.item for bi in items}            # type: ignore[union-attr]
        agg_scores: dict[str, list[float]] = defaultdict(list)
        per_domain: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for p in tqdm(predictions, desc="C score"):
            ci: ConditionalItem | None = items_by_id.get(p.item_id)
            if ci is None:
                continue
            pred = p.prediction if isinstance(p.prediction, dict) else {}
            level_paths: dict = pred.get("video_paths_by_level", {})
            variant_paths: list = pred.get("variant_video_paths", [])
            try:
                ref = _read_video(ci.reference_video_path)
            except Exception as exc:
                logger.warning("reference video missing for %s (%s)", ci.item_id, exc)
                continue

            level_videos = {}
            for lvl, path in level_paths.items():
                try:
                    level_videos[int(lvl)] = _read_video(path)
                except Exception as exc:
                    logger.warning("level video missing: %s (%s)", path, exc)
            if 0 not in level_videos:
                continue

            generated = level_videos[0]
            ssim_v = control.blur_ssim(generated, ref)
            edge = control.edge_f1(generated, ref)
            depth = control.depth_si_rmse(generated, ref)
            miou = control.mask_miou(generated, ref)
            visual = 0.5 * ssim_v + 0.5 * edge        # quick proxy visual-quality

            div = 0.0
            if len(variant_paths) >= 2:
                try:
                    variants = [_read_video(v) for v in variant_paths]
                    div = diversity.generation_diversity(variants)
                except Exception as exc:
                    logger.warning("variant load failed for %s (%s)", ci.item_id, exc)

            robust = control.robustness_score(
                level_videos, ref, primary_metric_fn=control.blur_ssim,
            )

            metrics = {
                "blur_ssim": ssim_v,
                "edge_f1": edge,
                "depth_si_rmse": depth,
                "mask_miou": miou,
                "visual_quality": visual,
                "generation_diversity": div,
                "robustness_score": robust,
            }
            for k, v in metrics.items():
                agg_scores[k].append(float(v))
            dom = ci.domain if isinstance(ci.domain, str) else ci.domain.value
            for k, v in metrics.items():
                per_domain[dom][k].append(float(v))

        scores = {k: float(np.mean(v)) if v else 0.0 for k, v in agg_scores.items()}
        per_domain_out = {
            d: {k: float(np.mean(v)) for k, v in inner.items()}
            for d, inner in per_domain.items()
        }
        return TrackScore(
            track=self.track_id,
            model_id=(predictions[0].model_id if predictions else "unknown"),
            n_items=len(items),
            scores=scores,
            per_domain=per_domain_out,
            per_physics_category={},
        )
