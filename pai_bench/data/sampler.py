"""Domain-stratified sampler.

# PAI-BENCH-2-CHANGE: enforces min_samples per domain and max_fraction caps
# from config/domains.yaml, preventing AV/robotics over-representation.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml

from pai_bench.data.schema import BenchmarkItem, Domain

logger = logging.getLogger(__name__)


def load_domain_config(config_path: Path) -> dict:
    return yaml.safe_load(Path(config_path).read_text())["domains"]


def stratified_sample(
    items: list[BenchmarkItem],
    domain_config: dict,
    target_total: int,
    seed: int = 42,
) -> list[BenchmarkItem]:
    rng = random.Random(seed)
    by_domain: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for bi in items:
        dom = getattr(bi.item, "domain", None)
        if dom is None:
            continue
        key = dom if isinstance(dom, str) else dom.value
        by_domain[key].append(bi)

    selected: list[BenchmarkItem] = []
    # First pass: enforce min_samples per domain (or all available if fewer exist).
    for dom, cfg in domain_config.items():
        pool = by_domain.get(dom, [])
        n_min = min(cfg.get("min_samples", 0), len(pool))
        rng.shuffle(pool)
        selected.extend(pool[:n_min])
        by_domain[dom] = pool[n_min:]

    # Second pass: top up toward target_total, respecting max_fraction caps.
    caps = {dom: int(cfg.get("max_fraction", 1.0) * target_total)
            for dom, cfg in domain_config.items()}
    counts = defaultdict(int)
    for bi in selected:
        dom = bi.item.domain                                # type: ignore[union-attr]
        counts[dom if isinstance(dom, str) else dom.value] += 1

    while len(selected) < target_total:
        progress = False
        for dom, pool in by_domain.items():
            if not pool:
                continue
            if counts[dom] >= caps.get(dom, target_total):
                continue
            selected.append(pool.pop())
            counts[dom] += 1
            progress = True
            if len(selected) >= target_total:
                break
        if not progress:
            logger.warning(
                "stratified_sample: exhausted pools at %d/%d items",
                len(selected), target_total,
            )
            break

    return selected
