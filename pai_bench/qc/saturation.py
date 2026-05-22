"""Saturation detection.

# PAI-BENCH-2-CHANGE: automated quarterly refresh signal absent from v1.
# Items where the top-3 models all exceed a threshold are scheduled for
# retirement in the next dataset refresh.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from pai_bench.data.schema import BenchmarkItem, QAItem


def detect_saturated_items(
    item_scores: dict[str, list[float]],
    top_n: int = 3,
    threshold: float = 0.90,
) -> list[str]:
    saturated: list[str] = []
    for item_id, scores in item_scores.items():
        if len(scores) < top_n:
            continue
        top = sorted(scores, reverse=True)[:top_n]
        if all(s >= threshold for s in top):
            saturated.append(item_id)
    return saturated


def saturation_report(
    items: Iterable[BenchmarkItem],
    item_scores: dict[str, list[float]],
    top_n: int = 3,
    threshold: float = 0.90,
) -> dict:
    saturated = set(detect_saturated_items(item_scores, top_n=top_n, threshold=threshold))
    by_domain: dict[str, list[str]] = defaultdict(list)
    by_physics: dict[str, list[str]] = defaultdict(list)
    for bi in items:
        if bi.item.item_id not in saturated:                # type: ignore[union-attr]
            continue
        dom = getattr(bi.item, "domain", None)
        phys = getattr(bi.item, "physics_category", None)
        if dom is not None:
            by_domain[dom if isinstance(dom, str) else dom.value].append(bi.item.item_id)
        if phys is not None:
            by_physics[phys if isinstance(phys, str) else phys.value].append(bi.item.item_id)
    return {
        "n_saturated": len(saturated),
        "saturation_by_domain": {k: len(v) for k, v in by_domain.items()},
        "saturation_by_physics_category": {k: len(v) for k, v in by_physics.items()},
        "recommended_replacements": len(saturated),
        "saturated_ids": sorted(saturated),
    }
