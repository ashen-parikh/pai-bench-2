"""Track CF: causal counterfactual reasoning.

# PAI-BENCH-2-CHANGE: entirely new track. Asks the model to predict how the
# scene's outcome changes when a single variable is intervened on. Scored
# via NLI entailment plus keyword-based direction/magnitude checks.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from pai_bench.data.schema import (
    BenchmarkItem, CounterfactualItem, ModelPrediction, TrackScore,
)
from pai_bench.metrics.counterfactual import counterfactual_delta_accuracy
from pai_bench.tracks.base import BaseTrack

logger = logging.getLogger(__name__)


class CounterfactualTrack(BaseTrack):
    track_id = "CF"

    def load_items(self, data_dir: Path) -> list[BenchmarkItem]:
        items_dir = Path(data_dir) / "items" / "CF"
        out: list[BenchmarkItem] = []
        for p in sorted(items_dir.glob("*.json")):
            payload = json.loads(p.read_text())
            cfi = CounterfactualItem(**payload)
            out.append(BenchmarkItem(track="CF", item=cfi))
        return out

    def evaluate(
        self,
        model_fn: Callable,
        items: list[BenchmarkItem],
    ) -> list[ModelPrediction]:
        preds: list[ModelPrediction] = []
        for bi in tqdm(items, desc="CF eval"):
            cfi: CounterfactualItem = bi.item               # type: ignore[assignment]
            question = (
                f"If {cfi.counterfactual_variable} were {cfi.counterfactual_change}, "
                f"how would the outcome change? Describe specifically."
            )
            response = model_fn({
                "video_path": cfi.base_video_path,
                "scenario": cfi.base_description,
                "question": question,
            })
            preds.append(ModelPrediction(
                model_id=response.get("model_id", "unknown"),
                item_id=cfi.item_id,
                track=self.track_id,
                prediction=str(response["text"]),
                confidence=response.get("confidence"),
                latency_ms=response.get("latency_ms"),
            ))
        return preds

    def score(
        self,
        predictions: list[ModelPrediction],
        items: list[BenchmarkItem],
    ) -> TrackScore:
        items_by_id = {bi.item.item_id: bi.item for bi in items}            # type: ignore[union-attr]
        ordered_items: list[CounterfactualItem] = []
        ordered_preds: list[str] = []
        per_domain: dict[str, list[int]] = defaultdict(list)
        for p in predictions:
            cfi: CounterfactualItem | None = items_by_id.get(p.item_id)
            if cfi is None:
                continue
            ordered_items.append(cfi)
            ordered_preds.append(p.prediction if isinstance(p.prediction, str) else str(p.prediction))

        metrics = counterfactual_delta_accuracy(ordered_preds, ordered_items)

        # Per-domain accuracy reuses the overall entailment result via item lookup.
        # Recompute on the fly for clarity.
        from pai_bench.metrics.counterfactual import _entailment_prob
        for cfi, pred in zip(ordered_items, ordered_preds):
            dom = cfi.domain if isinstance(cfi.domain, str) else cfi.domain.value
            ok = int(_entailment_prob(pred, cfi.counterfactual_outcome) >= 0.5)
            per_domain[dom].append(ok)

        per_domain_out = {
            d: {"accuracy": float(sum(v) / len(v)) if v else 0.0, "n": len(v)}
            for d, v in per_domain.items()
        }

        return TrackScore(
            track=self.track_id,
            model_id=(predictions[0].model_id if predictions else "unknown"),
            n_items=len(items),
            scores={
                "counterfactual_delta_accuracy": metrics["overall_accuracy"],
                "direction_accuracy": metrics["direction_accuracy"],
                "magnitude_accuracy": metrics["magnitude_accuracy"],
            },
            per_domain=per_domain_out,
            per_physics_category={
                cat: {"accuracy": info["accuracy"], "n": float(info["n"])}
                for cat, info in metrics["per_physics_category"].items()
            },
        )
