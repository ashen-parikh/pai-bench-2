"""Run a trivial deterministic baseline against the sample data.

Useful as a smoke test: confirms tracks load, predictions flow through
scoring, and the runner writes per-track JSON to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pai_bench.evaluation.runner import BenchmarkRunner

logging.basicConfig(level=logging.INFO)

ROOT = Path(__file__).resolve().parent.parent


def baseline_model_fn(request: dict) -> dict:
    """Always returns answer 0, an empty video path, or a stock CF text."""
    if "question" in request and "choices" in request:
        return {"model_id": "baseline", "answer_idx": 0, "confidence": 0.25}
    if "scenario" in request:
        return {
            "model_id": "baseline",
            "text": "The outcome would not change meaningfully.",
            "confidence": 0.25,
        }
    if "control_signals" in request:
        return {
            "model_id": "baseline",
            "video_paths_by_level": {},
            "variant_video_paths": [],
        }
    return {"model_id": "baseline", "video_path": ""}


def main() -> None:
    runner = BenchmarkRunner(
        config_dir=ROOT / "config",
        data_dir=ROOT / "data" / "sample",
    )
    scores = runner.run(
        model_id="baseline",
        model_fn=baseline_model_fn,
        output_dir=ROOT / "runs",
        tracks=["U", "CF"],
    )
    for tid, ts in scores.items():
        print(f"{tid}: {ts.scores}")


if __name__ == "__main__":
    main()
