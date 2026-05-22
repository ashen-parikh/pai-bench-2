"""Compute a human-baseline accuracy from annotation files.

Reads <data_dir>/human_baseline/<track>.json files of the form
  [{"item_id": str, "human_answer_idx": int}, ...]
and reports per-track / per-domain accuracy vs the ground-truth in
`<data_dir>/items/`. This is the number we publish alongside the
leaderboard as the practical ceiling.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pai_bench.data.loader import load_items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--track", default="U")
    args = p.parse_args()

    items = load_items(args.data_dir, args.track)
    items_by_id = {bi.item.item_id: bi.item for bi in items}        # type: ignore[union-attr]

    baseline_path = args.data_dir / "human_baseline" / f"{args.track}.json"
    if not baseline_path.exists():
        print(f"No human baseline at {baseline_path}; nothing to compute.")
        return

    rows = json.loads(baseline_path.read_text())
    correct: list[int] = []
    by_domain: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        item = items_by_id.get(row["item_id"])
        if item is None:
            continue
        ok = int(row["human_answer_idx"] == item.answer_idx)
        correct.append(ok)
        dom = item.domain if isinstance(item.domain, str) else item.domain.value
        by_domain[dom].append(ok)

    if not correct:
        print("No matching items.")
        return

    print(f"track={args.track}  n={len(correct)}  accuracy={sum(correct) / len(correct):.3f}")
    for dom, vals in by_domain.items():
        print(f"  {dom}: n={len(vals)} acc={sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
