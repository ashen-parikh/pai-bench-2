# PAI-Bench 2 — Track G

Physical AI benchmark, **unconditional video generation track**. Measures whether a video generative model has a working model of the physical world — does the video it produces actually obey physics — not just whether it produces plausible-looking pixels.

This README documents **Track G** only. The repository contains the broader benchmark scaffolding, but the public scoring story focuses on G.

License: Apache-2.0 · 37 tests · Python 3.10+

---

## What Track G measures

For each item the model is given a prompt (and a reference clip during annotation), and produces a video. PAI-Bench scores the output along two axes:

```
G_score = 0.30 · Quality_Score + 0.70 · Domain_Score
```

- **Quality Score (0.30)** — surface video quality: subject/background consistency, motion smoothness, aesthetic and imaging quality, video-text alignment.
- **Domain Score (0.70)** — *does the physics check out?* This is where v2's main contribution lives.

Domain Score dominates because v1's experience showed that high Quality Score correlates poorly with physical correctness — a Sora-style model can produce gorgeous footage with broken dynamics.

---

## What changed vs v1

Every change is greppable in the source via `PAI-BENCH-2-CHANGE:`.

### 1. Judge architecture: cross-vendor ensemble + analytic verifier

**v1** used a single MLLM judge (Qwen3-VL). One number out. No structure, no uncertainty.

**v2** uses `HybridJudge`, which runs in two stages:

```
                       ┌────────────────────────────────────┐
                       │       HybridJudge.score()          │
                       └─────────────────┬──────────────────┘
                                         │
                                         ▼
                       ┌────────────────────────────────────┐
                       │   PhysicsJudge (analytic verifier) │
                       │   handles RIGID_BODY, CONTACT,     │
                       │   FLUID (heuristic) via:           │
                       │     • per-category dynamics checks │
                       │     • 4 new analytic metrics ──────┼──── no MLLM ceiling
                       └─────────────────┬──────────────────┘
                                         │
                       ┌─────────────────┴──────────────────┐
                       │ tractable?                         │
                       └───┬────────────────────────────┬───┘
                          yes                           no
                           │                            ▼
                           │              ┌────────────────────────────────┐
                           │              │ EnsembleJudge (N MLLMs)        │
                           │              │   • median / mean / majority   │
                           │              │   • inter-judge std + range    │
                           │              │   • disagreement_flagged       │
                           │              │   • failed calls → score=None  │
                           │              │     (no false consensus)       │
                           │              └────────────┬───────────────────┘
                           │                           │
                           ▼                           ▼
                    return analytic              return ensemble
                    verdict + per-check          verdict + per-judge
                    breakdown + supp scores      breakdown + agreement
```

Every verdict carries an explicit `judge_type ∈ {analytic, heuristic, ensemble, none}` and `uncertainty ∈ {low, medium, high}`. The leaderboard MUST NOT aggregate items with `score=None` into the headline; they are reported as `n_unscored`.

### 2. Per-category analytic checks

When the item's `physics_category` is tractable, `PhysicsVerifier` runs category-specific pixel-physics tests instead of asking an MLLM.

**RIGID_BODY:**
| Check | Pass criterion |
|---|---|
| gravity_alignment | parabolic-fit acceleration sign matches `expected_physics.gravity_direction[1]` |
| collision_count | `\|detected − expected\| ≤ 1`; spikes = velocity jumps > `0.6 · median_v` |
| no_interpenetration | no centroid jump > `5 · median_step` |

**CONTACT:**
| Check | Pass criterion |
|---|---|
| friction_consistent | `mean(speed[T/2:]) ≤ 1.2 · mean(speed[:T/2])` |
| reflection_plausible | cosine between reflected-incoming and outgoing > 0.7 at sharpest bend |
| no_interpenetration | same teleport check |

**FLUID (heuristic, `uncertainty=medium`):**
| Check | Pass criterion |
|---|---|
| fluid_mass_conservation | `(max_area − min_area) / max_area < 0.5` |
| vorticity_plausible | `p99(\|curl\|) < 5.0` over Farneback flow |
| surface_smoothness | `std(canny_edge_density) < 20.0` |

Other categories (DEFORMABLE, THERMAL, ELECTROMAGNETIC) → `intractable` → ensemble fallback.

### 3. Four supplementary analytic metrics (no MLLM ceiling)

Run on every tractable item alongside the category-specific checks. Each returns a scalar in `[0, 1]` and has a permissive floor for pass/fail:

| Metric | Floor | What it catches | Implementation |
|---|---|---|---|
| `optical_flow_smoothness` | ≥ 0.30 | morphing, teleportation, judder | Farneback flow consecutive-field L2 delta → `1 / (1+x)` |
| `depth_stability` | ≥ 0.55 | depth flicker, shape-from-shading inconsistencies | DepthAnythingV2 per-frame depth, sample 256 random pixel pairs, measure fraction whose ordering stays constant |
| `motion_blob_count_stability` | ≥ 0.50 | phantom-object appearance / disappearance | adjacent-frame absdiff → threshold → connected components |
| `pose_validity` | ≥ 0.50 (or skipped) | impossible skeletons | MediaPipe pose, bone-length CoV across frames — returns `None` if no detector |

These are pixel physics, not MLLM judgments. They can't be gamed by a better-calibrated VLM.

### 4. Quality Score

`Quality_Score` is the mean of six sub-metrics (equal weight, pending v2 reweighting):

| Metric | Implementation |
|---|---|
| subject_consistency | DINO ViT-B/16 cosine-chain |
| background_consistency | CLIP ViT-B/32 cosine-chain |
| motion_smoothness | linear-midpoint interp residual (FILM/RIFE preferred) |
| aesthetic_quality | LAION predictor (1–10 normalised to 0–1) |
| imaging_quality | MUSIQ (`google/musiq-spaq`) |
| overall_consistency | ViCLIP video-text cosine |

> Open follow-up: drop ViCLIP from this average and add per-component weights to downweight `aesthetic_quality` so real-sensor footage isn't penalised for looking like real-sensor footage.

### 5. Annotation pipeline (drives per-item `expected_physics`)

`expected_physics` is a structured dict (gravity direction, expected collision count, etc.) that the analytic verifier reads. It is **human-authored**; the MLLM only assists with details. Pipeline:

```
candidate item
   ↓ MLLMAnnotator draft
   ↓ HumanReviewQueue (≥3 human annotations)
   ↓ filter_by_agreement (per-item majority ≥ 0.80)
   ↓
benchmark item ready for evaluation
```

---

## Install

```bash
git clone https://github.com/ashen-parikh/pai-bench-2.git
cd pai-bench-2
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

If you'll call MLLM judges for the ensemble fallback, also install PyAV and set keys:

```bash
.venv/bin/python -m pip install av
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
EOF
chmod 600 .env
```

`.env` is gitignored. Never commit either key.

---

## Quick start

### Compare v1 single-judge vs v2 hybrid on a real video

```bash
# downloads a CC BY 3.0 bouncing-ball clip from Wikimedia (1 MB)
mkdir -p data/sample/videos/external
curl -L -o data/sample/videos/external/Bouncing_Ball.webm \
  https://upload.wikimedia.org/wikipedia/commons/3/32/Bouncing_Ball.webm

.venv/bin/python scripts/demo_v1_vs_v2.py
```

This is the canonical demo. It:
- Decodes the real clip via PyAV
- Runs **v1** (single MLLMJudge) on the same pixels — one number out
- Runs **v2** (`HybridJudge` = `PhysicsJudge` + 3-Claude `EnsembleJudge` fallback) — per-check breakdown, four supplementary analytic scores, plus per-judge ensemble scores and inter-judge std/range/disagreement when the analytic side punts

### Run Track G against your own model

```bash
.venv/bin/pai-bench run \
  --model-id my-model \
  --model-config configs/my_model.yaml \
  --tracks G \
  --data-dir data/sample \
  --output-dir runs/my-model

.venv/bin/pai-bench score --run-dir runs/my-model --format both
```

Your `my_model.yaml` needs `model_fn: package.module:callable` pointing to a Python callable.

### Plugging in your VGM

`model_fn` is a black box. For Track G it receives `{"prompt": str, "item_id": str}` and returns `{"model_id": str, "video_path": str}`:

```python
# my_model.py
def my_model_fn(request: dict) -> dict:
    if "prompt" in request:
        out_path = my_vgm.generate(request["prompt"])      # your model
        return {"model_id": "my-vgm", "video_path": out_path}
    raise ValueError(f"unrecognised request: {request}")
```

Then your `my_model.yaml`:
```yaml
model_fn: my_model:my_model_fn
```

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
# 37 passed
```

Coverage relevant to Track G:
- All six Quality Score metrics return values in `[0, 1]`
- `PhysicsVerifier` returns the expected per-check verdict dict and supplementary scores
- `EnsembleJudge` aggregates correctly under median / mean / majority, drops failed calls, flags disagreement
- Physics metrics: clean-motion clips beat teleporting clips on flow smoothness; static-depth scenes beat flicker; single-object scenes beat chaos on blob stability

---

## Repo layout (G-relevant)

```
config/
  domains.yaml              # 8 domains with min_samples + max_fraction caps
  track_config.yaml         # G config: weights, ensemble models, analytic metric list
pai_bench/
  data/schema.py            # GenerationItem, ModelPrediction, TrackScore
  metrics/
    quality.py              # Quality Score (DINO, CLIP, MUSIQ, ViCLIP, ...)
    physics_verifier.py     # Per-category analytic dynamics checks
    physics_metrics.py      # NEW: optical_flow_smoothness, depth_stability,
                            #      motion_blob_count_stability, pose_validity
    domain.py               # Domain Score router (analytic ↔ MLLM)
  judge/
    base.py                 # BaseJudge interface
    physics_judge.py        # Wraps PhysicsVerifier
    mllm_judge.py           # OpenAI MLLM judge; returns score=None on failure
    anthropic_judge.py      # Anthropic Claude MLLM judge
    ensemble_judge.py       # N-MLLM ensemble + inter-judge agreement reporting
    hybrid_judge.py         # analytic → ensemble fallback
  tracks/
    generation.py           # Track G end-to-end
  evaluation/
    runner.py               # BenchmarkRunner (resumable per-track JSON cache)
    leaderboard.py          # G_score aggregation
    report.py               # JSON + Markdown writers
  cli.py                    # `pai-bench run|score`
scripts/
  demo_v1_vs_v2.py          # Canonical demo: single-MLLM vs hybrid on a real clip
  run_baseline_eval.py      # Constant-answer baseline smoke test
tests/                      # 37 tests
```

---

## Honest disclaimers

- **Analytic thresholds were tuned against synthetic clean clips.** `FLOW_SMOOTHNESS_FLOOR=0.30`, `DEPTH_STABILITY_FLOOR=0.55`, `BLOB_STABILITY_FLOOR=0.50` need recalibration against a held-out corpus of real, content-verified physics footage before they're usable for ranking. The Wikimedia bouncing-ball demo surfaced exactly this: the verdict failed three of four supplementary metrics even though the physics is correct, because real camera handling and lighting differ from synthetic test data.

- **Two slide-1 follow-ups are still open:**
  1. Drop ViCLIP from the Quality Score average and add per-component weights to downweight `aesthetic_quality`.
  2. Add per-2-second-window scoring + per-phenomenon temporal tagging (currently every quality metric returns a single scalar per clip).

- **Open architecture follow-up:** route the analytic verifier to the ensemble as a cross-check when the analytic score is marginal (`0.4 < score < 0.7`) or when its sub-check signals contradict (half PASS, half FAIL). Currently the routing is purely category-driven.

- **The MLLM ensemble is real but vendors charge.** A single run of a three-Claude ensemble on a 16-frame clip costs a few cents. Budget accordingly when scaling to the full G-track corpus.

---

## Acknowledgments

- Bouncing-ball reference clip via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bouncing_Ball.webm) (ScienceCOLA, CC BY 3.0). Downloaded separately at runtime; not committed.
- VBench-style quality metrics adapt prior work: subject/background consistency (DINO/CLIP cosine-chain), motion smoothness (FILM/RIFE-style interpolation residual), MUSIQ image quality, LAION aesthetic predictor.
- Depth via Depth-Anything-V2 (small) for analytic depth-stability checks.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).
