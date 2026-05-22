"""Abstract track interface.

Tracks share three stages: load items, run model, score predictions. Concrete
subclasses implement the stage-specific logic; the base class wires them
together in `run()` so the runner can treat tracks uniformly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from pai_bench.data.schema import BenchmarkItem, ModelPrediction, TrackScore

logger = logging.getLogger(__name__)


class BaseTrack(ABC):
    track_id: str = ""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = Path(config_dir) if config_dir else None

    @abstractmethod
    def load_items(self, data_dir: Path) -> list[BenchmarkItem]:
        """Load all items for this track."""

    @abstractmethod
    def evaluate(
        self,
        model_fn: Callable,
        items: list[BenchmarkItem],
    ) -> list[ModelPrediction]:
        """Run model on all items, return predictions."""

    @abstractmethod
    def score(
        self,
        predictions: list[ModelPrediction],
        items: list[BenchmarkItem],
    ) -> TrackScore:
        """Score predictions against ground truth."""

    def run(self, model_fn: Callable, data_dir: Path) -> TrackScore:
        items = self.load_items(Path(data_dir))
        logger.info("Track %s: loaded %d items", self.track_id, len(items))
        preds = self.evaluate(model_fn, items)
        return self.score(preds, items)
