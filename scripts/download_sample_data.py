"""Download the small sample dataset committed with the repo.

For now the sample data is already in `data/sample/`; this script verifies
its presence and prints a summary. When external blob storage is wired up
it will pull placeholder video files into `data/sample/videos/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"


def main() -> int:
    if not SAMPLE.exists():
        print(f"Sample data missing at {SAMPLE}", file=sys.stderr)
        return 1
    counts = {}
    for track_dir in (SAMPLE / "items").iterdir():
        counts[track_dir.name] = sum(1 for _ in track_dir.glob("*.json"))
    print("Sample dataset summary:")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
