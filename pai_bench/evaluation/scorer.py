"""Per-item to per-track aggregation helpers.

Kept thin: tracks already aggregate inside their `score()` methods. This
module exists so callers that want to rescore from raw predictions on disk
have a single entry point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pai_bench.data.loader import load_items
from pai_bench.data.schema import ModelPrediction, TrackScore
from pai_bench.tracks.base import BaseTrack
from pai_bench.tracks.conditional import ConditionalTrack
from pai_bench.tracks.counterfactual import CounterfactualTrack
from pai_bench.tracks.generation import GenerationTrack
from pai_bench.tracks.understanding import UnderstandingTrack

logger = logging.getLogger(__name__)

TRACK_CLASSES: dict[str, type[BaseTrack]] = {
    "G": GenerationTrack,
    "C": ConditionalTrack,
    "U": UnderstandingTrack,
    "CF": CounterfactualTrack,
}


def rescore_track(track_id: str, predictions_path: Path, data_dir: Path) -> TrackScore:
    if track_id not in TRACK_CLASSES:
        raise ValueError(f"unknown track {track_id}")
    cls = TRACK_CLASSES[track_id]
    track = cls()
    items = load_items(data_dir, track_id)
    preds_raw = json.loads(Path(predictions_path).read_text())
    preds = [ModelPrediction(**p) for p in preds_raw]
    return track.score(preds, items)
