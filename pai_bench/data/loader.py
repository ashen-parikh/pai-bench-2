"""Benchmark-item loader.

Loads JSON files under <data_dir>/items/<track>/ and returns typed
BenchmarkItem objects. Falls through to HuggingFace datasets if a
dataset identifier is provided.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pai_bench.data.schema import (
    BenchmarkItem, ConditionalItem, CounterfactualItem, GenerationItem, QAItem,
)

logger = logging.getLogger(__name__)

_SCHEMA = {
    "G": GenerationItem,
    "C": ConditionalItem,
    "U": QAItem,
    "CF": CounterfactualItem,
}


def load_items(data_dir: Path, track: str) -> list[BenchmarkItem]:
    if track not in _SCHEMA:
        raise ValueError(f"unknown track: {track}")
    items_dir = Path(data_dir) / "items" / track
    out: list[BenchmarkItem] = []
    for p in sorted(items_dir.glob("*.json")):
        payload = json.loads(p.read_text())
        cls = _SCHEMA[track]
        out.append(BenchmarkItem(track=track, item=cls(**payload)))
    logger.info("loaded %d items for track %s from %s", len(out), track, items_dir)
    return out


def load_from_hf(repo_id: str, track: str) -> list[BenchmarkItem]:
    from datasets import load_dataset
    ds = load_dataset(repo_id, split=track.lower())
    cls = _SCHEMA[track]
    return [BenchmarkItem(track=track, item=cls(**row)) for row in ds]
