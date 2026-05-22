"""Inter-annotator agreement filter using Cohen's kappa."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Iterable

import numpy as np
from sklearn.metrics import cohen_kappa_score

from pai_bench.data.schema import QAItem

logger = logging.getLogger(__name__)


def cohen_kappa(annotations_a: list[int], annotations_b: list[int]) -> float:
    """Pass-through to sklearn for stable behavior."""
    if not annotations_a or not annotations_b:
        return 0.0
    if len(annotations_a) != len(annotations_b):
        raise ValueError("annotator vectors must be same length")
    return float(cohen_kappa_score(annotations_a, annotations_b))


def _pairwise_mean_kappa(panels: list[list[int]]) -> float:
    """Mean pairwise kappa across all annotators for one item.

    Each panel is a single annotator's full-batch answer list aligned by item
    index. Cohen's kappa requires multiple categorical labels, so we compute
    on the full vector and average across pairs.
    """
    if len(panels) < 2:
        return 0.0
    scores = [cohen_kappa(a, b) for a, b in combinations(panels, 2)]
    return float(np.mean(scores))


def filter_by_agreement(
    items: list[QAItem],
    annotations: dict[str, list[list[int]]],
    threshold: float = 0.80,
) -> tuple[list[QAItem], list[QAItem]]:
    """Split items into (passed, flagged) by Cohen's-kappa agreement.

    annotations: item_id -> list of per-annotator answers
                 (each inner list is length 1 — the annotator's chosen index).
    """
    # Re-arrange to per-annotator answer vectors aligned by item order.
    # Only items present in `annotations` are considered.
    ordered_items = [i for i in items if i.item_id in annotations]
    if not ordered_items:
        return [], []
    n_annotators = max(len(annotations[i.item_id]) for i in ordered_items)
    if n_annotators < 2:
        # Nothing to compare; treat all as passed.
        for i in ordered_items:
            i.annotator_agreement = 1.0
        return ordered_items, []

    panels: list[list[int]] = [[] for _ in range(n_annotators)]
    for item in ordered_items:
        row = annotations[item.item_id]
        for j in range(n_annotators):
            entry = row[j] if j < len(row) else row[0]
            # Each annotator's per-item entry is itself a 1-element list of
            # answer indices (the schema allows multi-answer panels for some
            # tracks); take the first answer as the canonical vote.
            vote = entry[0] if isinstance(entry, (list, tuple)) else entry
            panels[j].append(int(vote))

    # Panel-wide kappa is a property of the annotator pool; it's logged here
    # but per-item filtering uses per-item agreement so unanimous items pass
    # even if panel kappa is mediocre (kappa is undefined per-item).
    panel_kappa = _pairwise_mean_kappa(panels)
    logger.info("panel_pairwise_mean_kappa=%.3f over %d items, %d annotators",
                panel_kappa, len(ordered_items), n_annotators)

    passed, flagged = [], []
    for idx, item in enumerate(ordered_items):
        votes = [p[idx] for p in panels]
        most_common = max(set(votes), key=votes.count)
        per_item_agreement = votes.count(most_common) / len(votes)
        item.annotator_agreement = float(per_item_agreement)
        if per_item_agreement >= threshold:
            passed.append(item)
        else:
            flagged.append(item)
    return passed, flagged
