"""Track U: video understanding (MLLM QA).

# PAI-BENCH-2-CHANGE: scaled to 1800 items across 8 domains with 35%
# multi-hop chains; multihop_chain_accuracy and error_propagation_rate
# are first-class metrics.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from pai_bench.data.schema import BenchmarkItem, ModelPrediction, QAItem, TrackScore
from pai_bench.tracks.base import BaseTrack

logger = logging.getLogger(__name__)

# Language prior threshold mirrors track_config.yaml.
LANGUAGE_PRIOR_THRESHOLD = 0.6


class UnderstandingTrack(BaseTrack):
    track_id = "U"

    def load_items(self, data_dir: Path) -> list[BenchmarkItem]:
        items_dir = Path(data_dir) / "items" / "U"
        out: list[BenchmarkItem] = []
        for p in sorted(items_dir.glob("*.json")):
            payload = json.loads(p.read_text())
            qa = QAItem(**payload)
            out.append(BenchmarkItem(track="U", item=qa))
        return out

    def evaluate(
        self,
        model_fn: Callable,
        items: list[BenchmarkItem],
    ) -> list[ModelPrediction]:
        preds: list[ModelPrediction] = []
        for bi in tqdm(items, desc="U eval"):
            qa: QAItem = bi.item                # type: ignore[assignment]
            request = {
                "video_path": qa.video_path,
                "question": qa.question,
                "choices": qa.choices,
            }
            response = model_fn(request)
            preds.append(ModelPrediction(
                model_id=response.get("model_id", "unknown"),
                item_id=qa.item_id,
                track=self.track_id,
                prediction={"answer_idx": int(response["answer_idx"])},
                confidence=response.get("confidence"),
                latency_ms=response.get("latency_ms"),
            ))
        return preds

    def score(
        self,
        predictions: list[ModelPrediction],
        items: list[BenchmarkItem],
    ) -> TrackScore:
        # Map item_id -> (QAItem, prediction).
        by_id: dict[str, tuple[QAItem, ModelPrediction]] = {}
        for bi in items:
            qa: QAItem = bi.item                # type: ignore[assignment]
            if qa.language_prior_score >= LANGUAGE_PRIOR_THRESHOLD:
                logger.warning(
                    "item %s has language_prior_score=%.2f >= %.2f; should have "
                    "been filtered by language_prior auditor.",
                    qa.item_id, qa.language_prior_score, LANGUAGE_PRIOR_THRESHOLD,
                )
            by_id[qa.item_id] = (qa, None)      # type: ignore[assignment]
        for p in predictions:
            qa, _ = by_id.get(p.item_id, (None, None))
            if qa is not None:
                by_id[p.item_id] = (qa, p)

        # Overall and per-domain accuracy.
        correct = []
        per_domain: dict[str, list[int]] = defaultdict(list)
        for qa, p in by_id.values():
            if p is None:
                continue
            pred_idx = int(p.prediction["answer_idx"]) if isinstance(p.prediction, dict) else -1
            ok = int(pred_idx == qa.answer_idx)
            correct.append(ok)
            dom = qa.domain if isinstance(qa.domain, str) else qa.domain.value
            per_domain[dom].append(ok)

        accuracy = float(np.mean(correct)) if correct else 0.0
        per_domain_acc = {d: {"accuracy": float(np.mean(v)), "n": len(v)} for d, v in per_domain.items()}

        # Multi-hop chain accuracy and error propagation rate.
        chains: dict[str, list[tuple[QAItem, ModelPrediction | None]]] = defaultdict(list)
        for qa, p in by_id.values():
            if qa.chain_id:
                chains[qa.chain_id].append((qa, p))

        chain_correct = 0
        chain_total = 0
        first_error_hops: list[int] = []
        for chain_id, members in chains.items():
            members.sort(key=lambda x: x[0].hop_index or 0)
            chain_total += 1
            chain_ok = True
            for qa, p in members:
                if p is None:
                    chain_ok = False
                    first_error_hops.append(qa.hop_index or 0)
                    break
                pred_idx = int(p.prediction["answer_idx"]) if isinstance(p.prediction, dict) else -1
                if pred_idx != qa.answer_idx:
                    chain_ok = False
                    first_error_hops.append(qa.hop_index or 0)
                    break
            if chain_ok:
                chain_correct += 1

        multihop_chain_accuracy = float(chain_correct / chain_total) if chain_total else 0.0
        # Error propagation rate: fraction of incorrect chains whose first error
        # was at hop > 0 (i.e. a downstream-hop failure rather than initial-step).
        if first_error_hops:
            downstream = sum(1 for h in first_error_hops if h > 0)
            error_propagation_rate = float(downstream / len(first_error_hops))
            mean_first_error_hop = float(np.mean(first_error_hops))
        else:
            error_propagation_rate = 0.0
            mean_first_error_hop = 0.0

        return TrackScore(
            track=self.track_id,
            model_id=(predictions[0].model_id if predictions else "unknown"),
            n_items=len(items),
            scores={
                "accuracy": accuracy,
                "multihop_chain_accuracy": multihop_chain_accuracy,
                "error_propagation_rate": error_propagation_rate,
                "mean_first_error_hop": mean_first_error_hop,
                "n_chains": float(chain_total),
            },
            per_domain=per_domain_acc,
            per_physics_category={},
        )
