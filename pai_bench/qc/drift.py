"""Benchmark difficulty drift tracker.

Watches the average PAI-Index across submitted models over time. A rising
mean without new items signals the benchmark is drifting toward saturation
and needs a refresh.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Iterable

import numpy as np

QUARTER_DAYS = 91


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s)


def benchmark_difficulty_over_time(
    historical_scores: list[dict],
    drift_threshold: float = 0.05,
) -> dict:
    """Aggregate model PAI-Index by quarter and report drift."""
    if not historical_scores:
        return {"trend": [], "drift_per_quarter": 0.0, "recommendation": "insufficient_data"}

    quarters: dict[str, list[float]] = defaultdict(list)
    for entry in historical_scores:
        d = _parse_date(entry["date"])
        # Quarter label like "2026Q1".
        q = f"{d.year}Q{(d.month - 1) // 3 + 1}"
        quarters[q].append(float(entry["pai_index"]))

    sorted_q = sorted(quarters)
    trend = [(q, float(np.mean(quarters[q]))) for q in sorted_q]
    drift_per_quarter = 0.0
    if len(trend) >= 2:
        means = [m for _, m in trend]
        drift_per_quarter = float(np.mean(np.diff(means)))

    if drift_per_quarter > drift_threshold:
        recommendation = "refresh_items"
    elif drift_per_quarter < -drift_threshold:
        recommendation = "investigate_regression"
    else:
        recommendation = "stable"

    return {
        "trend": trend,
        "drift_per_quarter": drift_per_quarter,
        "recommendation": recommendation,
    }
