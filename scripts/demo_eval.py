"""End-to-end demo against sample data.

Defines two stub models — one that nearly always picks answer 0 (poor),
another that nails U items and gives partly-correct CF text (decent) —
then runs them through tracks U and CF and prints the resulting scores
plus the PAI-Index sub-components for inspection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pai_bench.data.loader import load_items
from pai_bench.evaluation.leaderboard import Leaderboard, pai_index
from pai_bench.evaluation.runner import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "sample"
RUNS_DIR = ROOT / "runs"


# Lookup tables built from the sample data so models can be deterministic.
_U_ANSWERS = {bi.item.item_id: bi.item.answer_idx for bi in load_items(DATA_DIR, "U")}
_CF_KEYWORDS = {
    bi.item.item_id: (bi.item.direction_keyword, bi.item.magnitude_bin)
    for bi in load_items(DATA_DIR, "CF")
}


def stub_constant(request: dict) -> dict:
    """Always picks index 0 for QA; says nothing useful for CF."""
    if "question" in request and "choices" in request:
        return {"model_id": "constant", "answer_idx": 0, "confidence": 0.25}
    return {
        "model_id": "constant",
        "text": "The outcome would be similar.",
        "confidence": 0.25,
    }


def stub_oracle(request: dict) -> dict:
    """Always correct on U; for CF emits a templated direction+magnitude phrase."""
    if "question" in request and "choices" in request:
        # The sample data carries item-ids implicitly via the video_path; we
        # cheat for the demo by using the path's basename as the id key.
        # (In a real run the model has no such oracle.)
        item_id = Path(request["video_path"]).stem
        # Map sample video filenames back to item ids.
        item_id_map = {
            "u_001_av": "u_001_av_action",
            "u_002_robot": "u_002_robotics_grasp",
            "u_003": "u_003_bad_language_prior",
            "u_004": "u_004_low_agreement",
            "u_005": "u_005_chain_hop0",       # ambiguous — chain shares one video
        }
        return {
            "model_id": "oracle",
            "answer_idx": _U_ANSWERS.get(item_id_map.get(item_id, item_id), 0),
            "confidence": 0.95,
        }
    # CF response: use the per-item direction + magnitude bin to compose a phrase.
    item_id = Path(request["video_path"]).stem
    item_id_map = {f"cf_00{i}": f"cf_00{i}_{suffix}"
                   for i, suffix in enumerate(["rigid", "fluid", "contact", "deformable", "rigid"], start=1)}
    direction, magnitude = _CF_KEYWORDS.get(item_id_map.get(item_id, item_id), (None, None))
    text = f"It would be {magnitude or 'much more'} and {direction or 'slower'}."
    return {"model_id": "oracle", "text": text, "confidence": 0.95}


def main() -> None:
    runner = BenchmarkRunner(config_dir=ROOT / "config", data_dir=DATA_DIR)

    constant_scores = runner.run(
        "constant_model", stub_constant, RUNS_DIR, tracks=["U", "CF"], resume=False,
    )
    oracle_scores = runner.run(
        "oracle_model", stub_oracle, RUNS_DIR, tracks=["U", "CF"], resume=False,
    )

    lb = Leaderboard()
    lb.add_model("constant_model", constant_scores)
    lb.add_model("oracle_model", oracle_scores)
    df = lb.rank()
    print("\n=== leaderboard ===")
    print(df.to_string(index=False))

    print("\n=== per-track for oracle_model ===")
    for tid, ts in oracle_scores.items():
        print(f"[{tid}] n={ts.n_items} scores={json.dumps(ts.scores, indent=2)}")
        if ts.per_domain:
            print(f"    per_domain={ts.per_domain}")

    print(f"\nconstant PAI-Index={pai_index(constant_scores):.4f}")
    print(f"oracle   PAI-Index={pai_index(oracle_scores):.4f}")


if __name__ == "__main__":
    main()
