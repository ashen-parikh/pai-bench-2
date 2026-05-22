"""Track-level tests focused on aggregation correctness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pai_bench.data.schema import (
    BenchmarkItem, CounterfactualItem, Domain, ModelPrediction,
    PhysicsCategory, QAItem,
)
from pai_bench.tracks.counterfactual import CounterfactualTrack
from pai_bench.tracks.understanding import UnderstandingTrack

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"


def _qa(item_id: str, answer_idx: int, chain_id: str | None = None,
        hop: int | None = None) -> QAItem:
    return QAItem(
        item_id=item_id,
        video_path="/tmp/x",
        domain=Domain.EVERYDAY_PHYSICS,
        physics_category=PhysicsCategory.RIGID_BODY,
        question="?",
        choices=["a", "b", "c", "d"],
        answer_idx=answer_idx,
        requires_temporal=True,
        language_prior_score=0.0,
        annotator_agreement=0.9,
        chain_id=chain_id,
        hop_index=hop,
    )


def _pred(item_id: str, answer_idx: int, track: str = "U") -> ModelPrediction:
    return ModelPrediction(
        model_id="m", item_id=item_id, track=track,
        prediction={"answer_idx": answer_idx},
    )


def test_understanding_chain_accuracy_full_chain_correct():
    items = [
        BenchmarkItem(track="U", item=_qa("a", 0, "c1", 0)),
        BenchmarkItem(track="U", item=_qa("b", 1, "c1", 1)),
        BenchmarkItem(track="U", item=_qa("c", 2, "c1", 2)),
    ]
    preds = [_pred("a", 0), _pred("b", 1), _pred("c", 2)]
    ts = UnderstandingTrack().score(preds, items)
    assert ts.scores["multihop_chain_accuracy"] == pytest.approx(1.0)
    assert ts.scores["accuracy"] == pytest.approx(1.0)


def test_understanding_chain_accuracy_one_wrong_invalidates_chain():
    items = [
        BenchmarkItem(track="U", item=_qa("a", 0, "c1", 0)),
        BenchmarkItem(track="U", item=_qa("b", 1, "c1", 1)),
        BenchmarkItem(track="U", item=_qa("c", 2, "c1", 2)),
    ]
    preds = [_pred("a", 0), _pred("b", 0), _pred("c", 2)]   # b wrong
    ts = UnderstandingTrack().score(preds, items)
    assert ts.scores["multihop_chain_accuracy"] == pytest.approx(0.0)
    assert ts.scores["accuracy"] == pytest.approx(2 / 3)
    # First error at hop 1 -> error_propagation_rate == 1.0 (single chain).
    assert ts.scores["mean_first_error_hop"] == pytest.approx(1.0)
    assert ts.scores["error_propagation_rate"] == pytest.approx(1.0)


def test_counterfactual_direction_accuracy_keyword_match():
    item = CounterfactualItem(
        item_id="cf1",
        base_video_path="/tmp/x",
        base_description="b",
        counterfactual_variable="v",
        counterfactual_change="doubled",
        base_outcome="x",
        counterfactual_outcome="The cylinder fills more slowly.",
        domain=Domain.LABORATORY,
        physics_category=PhysicsCategory.FLUID,
        direction_keyword="slower",
        magnitude_bin="much_less",
    )
    items = [BenchmarkItem(track="CF", item=item)]
    preds = [ModelPrediction(model_id="m", item_id="cf1", track="CF",
                             prediction="It fills much less and slower than before.")]
    ts = CounterfactualTrack().score(preds, items)
    assert ts.scores["direction_accuracy"] == pytest.approx(1.0)
    assert ts.scores["magnitude_accuracy"] == pytest.approx(1.0)


def test_sample_dataset_loads():
    """Sanity check that the bundled sample data validates against the schema."""
    track_u = UnderstandingTrack().load_items(SAMPLE)
    assert any(bi.item.item_id == "u_001_av_action" for bi in track_u)
    track_cf = CounterfactualTrack().load_items(SAMPLE)
    assert len(track_cf) >= 3


def test_runner_resume_loads_cached_track(tmp_path: Path):
    """Verify resume() reads on-disk TrackScore JSON without re-running."""
    from pai_bench.evaluation.runner import BenchmarkRunner
    from pai_bench.data.schema import TrackScore

    run_dir = tmp_path / "model_x"
    run_dir.mkdir(parents=True)
    cached = TrackScore(
        track="U", model_id="model_x", n_items=3,
        scores={"accuracy": 0.75},
    )
    (run_dir / "U.json").write_text(cached.model_dump_json())
    runner = BenchmarkRunner(
        config_dir=Path(__file__).resolve().parent.parent / "config",
        data_dir=SAMPLE,
    )
    resumed = runner.resume("model_x", tmp_path)
    assert "U" in resumed
    assert resumed["U"].scores["accuracy"] == pytest.approx(0.75)
