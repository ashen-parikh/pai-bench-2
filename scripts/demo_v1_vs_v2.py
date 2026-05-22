"""Run a real downloaded video through both the v1 and v2 G-track judges.

Video: Bouncing Ball (Wikimedia Commons, CC BY 3.0, ScienceCOLA)
       https://commons.wikimedia.org/wiki/File:Bouncing_Ball.webm
       26s @ 25fps, 640x480. Real footage of a ball bouncing on a surface.

v1 pipeline (single judge, no structure)
    MLLMJudge — one MLLM emits a single 0-1 score with optional rationale.
    With no OPENAI_API_KEY it returns the neutral 0.5 fallback. That fallback
    is itself a v1 failure mode: the system has no way to signal "I don't
    actually know" beyond the rationale string.

v2 pipeline (HybridJudge)
    1. PhysicsJudge — analytic verifier. For RIGID_BODY / CONTACT it returns
       a per-check pass/fail dict (gravity alignment, collision count,
       interpenetration) plus the four new physics_metrics scores
       (optical_flow_smoothness, depth_stability, motion_blob_count_stability,
       pose_validity). All bound by pixel physics, not by any MLLM.
    2. EnsembleJudge — only consulted when the analytic verifier returns
       intractable. Aggregates N MLLMs and reports inter-judge std/range so
       a low-consensus verdict is visibly low-consensus rather than
       laundered through a single point estimate.

We run the same clip under TWO physics_category assignments so you see both
branches:
    pass 1 — RIGID_BODY  → analytic verifier handles it
    pass 2 — THERMAL     → intractable, ensemble fallback runs
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from pai_bench.data.schema import Domain, GenerationItem, PhysicsCategory
from pai_bench.judge import (
    AnthropicMLLMJudge, EnsembleJudge, HybridJudge, MLLMJudge, PhysicsJudge,
)
from pai_bench.judge.base import BaseJudge

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

VIDEO_PATH = Path(__file__).resolve().parent.parent / "data/sample/videos/external/Bouncing_Ball.webm"
TARGET_FRAMES = 16      # decimate so the demo runs in seconds and per-call upload stays small


def load_video(path: Path, n_frames: int) -> np.ndarray:
    """Decode the file, evenly subsample n_frames, return (T,H,W,3) float32 in [0,1]."""
    raw = iio.imread(path, plugin="pyav")
    print(f"[decode] {path.name}: full shape={raw.shape}, dtype={raw.dtype}")
    if raw.ndim != 4:
        raise ValueError(f"expected (T,H,W,3); got {raw.shape}")
    idx = np.linspace(0, raw.shape[0] - 1, n_frames).astype(int)
    sampled = raw[idx].astype(np.float32) / 255.0
    print(f"[decode] subsampled to {sampled.shape}")
    return sampled


# Cross-vendor 3-judge ensemble. Anthropic models when ANTHROPIC_API_KEY is
# set; gpt-4o-mini joins if OPENAI_API_KEY is set too. Mixing vendors reduces
# the chance the median is dragged by a single vendor's training-data quirks.
ANTHROPIC_MODELS = (
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    import os
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def build_item(category: PhysicsCategory, item_id: str) -> GenerationItem:
    return GenerationItem(
        item_id=item_id,
        prompt="A ball bounces on a flat surface, losing height with each bounce.",
        reference_video_path=str(VIDEO_PATH),
        domain=Domain.EVERYDAY_PHYSICS,
        physics_category=category,
        qa_pairs=[],
        expected_physics={"gravity_direction": [0.0, 1.0], "expected_collisions": 3},
    )


def make_v2_judge() -> HybridJudge:
    import os
    members: list[BaseJudge] = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        members.extend(AnthropicMLLMJudge(model=m) for m in ANTHROPIC_MODELS)
    if os.environ.get("OPENAI_API_KEY"):
        members.append(MLLMJudge(model="gpt-4o-mini"))
    if not members:
        raise SystemExit(
            "Set ANTHROPIC_API_KEY (and/or OPENAI_API_KEY) in .env before running."
        )
    print(f"[ensemble] {len(members)} members: "
          f"{[getattr(j, 'model', getattr(j, 'judge_type', '?')) for j in members]}")
    ensemble = EnsembleJudge(
        judges=members,
        aggregation="median",
        disagreement_threshold=0.15,
    )
    return HybridJudge(physics_judge=PhysicsJudge(), mllm_judge=ensemble)


def fmt(x):
    return "  None" if x is None else f"{x:6.3f}"


def print_v1(video: np.ndarray, item: GenerationItem) -> dict:
    """v1 architecture: one single MLLM judge, no analytic side, no ensemble."""
    import os
    print(f"\n=== v1: single MLLM judge (no analytic side) ===")
    if os.environ.get("ANTHROPIC_API_KEY"):
        judge: BaseJudge = AnthropicMLLMJudge(model="claude-sonnet-4-5")
    elif os.environ.get("OPENAI_API_KEY"):
        judge = MLLMJudge(model="gpt-4o-mini")
    else:
        raise SystemExit("v1 requires at least one of ANTHROPIC_API_KEY / OPENAI_API_KEY")
    t0 = time.perf_counter()
    out = judge.score(video, item)
    elapsed = time.perf_counter() - t0
    print(f"  judge_type   : {out.get('judge_type')}")
    print(f"  uncertainty  : {out.get('uncertainty')}")
    print(f"  score        : {fmt(out.get('score'))}")
    print(f"  violations   : {out.get('violations')}")
    print(f"  rationale    : {out.get('rationale')!r}")
    print(f"  elapsed      : {elapsed:.2f}s")
    return out


def print_v2(video: np.ndarray, item: GenerationItem) -> dict:
    print(f"\n=== v2: HybridJudge — physics_category={item.physics_category} ===")
    judge = make_v2_judge()
    t0 = time.perf_counter()
    out = judge.score(video, item)
    elapsed = time.perf_counter() - t0
    print(f"  judge_type    : {out.get('judge_type')}")
    print(f"  verifier_type : {out.get('verifier_type')}")
    print(f"  uncertainty   : {out.get('uncertainty')}")
    print(f"  score         : {fmt(out.get('score'))}")
    if "passed" in out:
        print(f"  passed        : {out['passed']}")
    print(f"  elapsed       : {elapsed:.2f}s")
    if out.get("checks"):
        print(f"  per-check verdicts:")
        for name, ok in out["checks"].items():
            mark = "PASS" if ok else "FAIL"
            print(f"    {name:32s} {mark}")
    if out.get("supplementary_scores"):
        print(f"  new analytic metrics (physics_metrics.py):")
        for name, val in out["supplementary_scores"].items():
            print(f"    {name:32s} {fmt(val)}")
    if "per_judge_scores" in out and out["per_judge_scores"]:
        print(f"  ensemble inter-judge:")
        for member in out.get("per_judge", []):
            print(f"    {member['judge_type']:24s} score={fmt(member['score'])}")
        print(f"    inter_judge_std         : {fmt(out.get('inter_judge_std'))}")
        print(f"    inter_judge_range       : {fmt(out.get('inter_judge_range'))}")
        print(f"    disagreement_flagged    : {out.get('disagreement_flagged')}")
    if out.get("violations"):
        print(f"  violations    : {out['violations']}")
    return out


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not VIDEO_PATH.exists():
        raise SystemExit(
            f"video not found: {VIDEO_PATH}\n"
            "Run: curl -L -o data/sample/videos/external/Bouncing_Ball.webm "
            "https://upload.wikimedia.org/wikipedia/commons/3/32/Bouncing_Ball.webm"
        )

    video = load_video(VIDEO_PATH, TARGET_FRAMES)

    print("\n" + "=" * 70)
    print("PASS 1: tractable physics (RIGID_BODY) — analytic verifier handles it")
    print("=" * 70)
    item_rb = build_item(PhysicsCategory.RIGID_BODY, "wm_bball_rigid")
    v1_rb = print_v1(video, item_rb)
    v2_rb = print_v2(video, item_rb)

    print("\n" + "=" * 70)
    print("PASS 2: intractable physics (THERMAL) — ensemble fallback runs")
    print("=" * 70)
    item_th = build_item(PhysicsCategory.THERMAL, "wm_bball_thermal")
    v1_th = print_v1(video, item_th)
    v2_th = print_v2(video, item_th)

    print("\n=== one-line summary ===")
    print(f"PASS 1 RIGID_BODY  v1={fmt(v1_rb.get('score'))}  v2={fmt(v2_rb.get('score'))}  ({v2_rb.get('verifier_type')})")
    print(f"PASS 2 THERMAL     v1={fmt(v1_th.get('score'))}  v2={fmt(v2_th.get('score'))}  ({v2_th.get('verifier_type')})")


if __name__ == "__main__":
    main()
