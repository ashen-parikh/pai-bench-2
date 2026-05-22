"""Expert-audit sampler.

For each refresh cycle, sample 20% of items stratified by domain and
physics category and route them to expert reviewers. The sampler is
deterministic given a seed so audit panels can be re-created.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from pai_bench.data.schema import BenchmarkItem


def expert_audit_sample(
    items: Iterable[BenchmarkItem],
    fraction: float = 0.20,
    seed: int = 0,
) -> list[BenchmarkItem]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rng = random.Random(seed)
    by_strata: dict[tuple[str, str], list[BenchmarkItem]] = defaultdict(list)
    for bi in items:
        dom = getattr(bi.item, "domain", None)
        phys = getattr(bi.item, "physics_category", None)
        key = (
            dom if isinstance(dom, str) else (dom.value if dom else "unknown"),
            phys if isinstance(phys, str) else (phys.value if phys else "unknown"),
        )
        by_strata[key].append(bi)
    out: list[BenchmarkItem] = []
    for strata, members in by_strata.items():
        rng.shuffle(members)
        k = max(1, int(round(len(members) * fraction)))
        out.extend(members[:k])
    return out
