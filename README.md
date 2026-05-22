# PAI-Bench 2

Physical AI Benchmark for Video Generative Models (VGMs) and Multimodal LLMs (MLLMs).

## Tracks

- **G**  - Unconditional generation: does generated video obey physics?
- **C**  - Conditional generation: does the model follow control signals?
- **U**  - Video understanding: can the model reason about physical scenes?
- **CF** - Causal counterfactual: does the model have a causal world model? (new in v2)
- **DV** - Downstream validity: do scores predict real task performance? (new in v2, maintainer-held)

## Install

```bash
pip install -e .
```

## Quick start

```bash
pai-bench run --model-id my-model --model-config configs/my_model.yaml \
              --tracks G,C,U,CF \
              --data-dir data/sample \
              --output-dir runs/my-model

pai-bench score --run-dir runs/my-model --format both
pai-bench leaderboard --results-dir runs/ --output leaderboard.md
```

## What changed from v1

Look for `# PAI-BENCH-2-CHANGE:` tags across the codebase. Key changes:

1. **Physics verifier replaces MLLM-as-Judge** for tractable scenarios (rigid body, contact). MLLM judge remains as fallback with explicit uncertainty flag.
2. **Robustness curve** in track C: control signals are degraded across 4 levels, fidelity falloff AUC reported.
3. **Multi-hop reasoning** in track U: 35% of items are multi-hop chains; chain accuracy and error propagation rate are first-class metrics.
4. **Counterfactual track (CF)** tests causal world model rather than static scene description.
5. **Downstream validity track (DV)** validates that benchmark scores predict real task performance (maintainer-controlled, held-out).
6. **Domain stratification** with hard min_samples floors and max_fraction caps across 8 domains.
7. **Saturation QC** automatically retires items the top-3 models exceed 90% on.
8. **PAI-Index** weights U and CF more heavily than G/C.
