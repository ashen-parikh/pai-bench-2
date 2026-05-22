"""Unit tests for metric implementations.

These avoid downloading the heavy DINO/CLIP/MUSIQ models by relying on the
fallback codepaths the metric module exposes when the underlying models
are unavailable. Each test focuses on input validation, output range, and
formula correctness against a hand-checked reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from pai_bench.annotation.irt_calibration import IRTCalibrator
from pai_bench.annotation.itr_filter import cohen_kappa, filter_by_agreement
from pai_bench.annotation.language_prior import LanguagePriorAuditor
from pai_bench.data.schema import (
    Domain, GenerationItem, PhysicsCategory, QAItem,
)
from pai_bench.metrics import control, quality
from pai_bench.metrics.physics_verifier import PhysicsVerifier
from pai_bench.qc.saturation import detect_saturated_items


def _dummy_video(t: int = 8, h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((t, h, w, 3)).astype(np.float32)


def test_subject_consistency_in_unit_range():
    v = _dummy_video()
    score = quality.subject_consistency(v)
    assert 0.0 <= score <= 1.0


def test_edge_f1_perfect_match_is_one():
    rng = np.random.default_rng(0)
    v = rng.random((4, 32, 32, 3)).astype(np.float32)
    score = control.edge_f1(v, v)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_blur_ssim_perfect_match_is_one():
    rng = np.random.default_rng(0)
    v = rng.random((4, 32, 32, 3)).astype(np.float32)
    score = control.blur_ssim(v, v)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0, abs=1e-3)


def test_cohen_kappa_matches_sklearn():
    a = [0, 1, 1, 0, 2, 1]
    b = [0, 1, 0, 0, 2, 1]
    from sklearn.metrics import cohen_kappa_score
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_irt_fit_returns_correct_shapes():
    rng = np.random.default_rng(0)
    responses = (rng.random((6, 10)) > 0.4).astype(int)
    cal = IRTCalibrator(max_iter=40)
    a, b = cal.fit(responses)
    assert a.shape == (10,)
    assert b.shape == (10,)


def test_language_prior_rejects_above_threshold():
    items = [
        QAItem(
            item_id="i1", video_path="/tmp/x", domain=Domain.EVERYDAY_PHYSICS,
            physics_category=PhysicsCategory.RIGID_BODY,
            question="?", choices=["a", "b", "c", "d"], answer_idx=0,
            requires_temporal=True, language_prior_score=0.0,
            annotator_agreement=0.95,
        ),
        QAItem(
            item_id="i2", video_path="/tmp/y", domain=Domain.EVERYDAY_PHYSICS,
            physics_category=PhysicsCategory.RIGID_BODY,
            question="?", choices=["a", "b", "c", "d"], answer_idx=0,
            requires_temporal=True, language_prior_score=0.0,
            annotator_agreement=0.95,
        ),
    ]

    def stub(question: str, choices: list[str]) -> int:
        # First item: always correct (will be rejected). Second: always wrong.
        return 0 if "easy" in question else 3

    items[0].question = "easy"
    items[1].question = "hard"
    auditor = LanguagePriorAuditor(threshold=0.6, client_fn=stub)
    passed = auditor.audit_batch(items, n_trials=4)
    assert [i.item_id for i in passed] == ["i2"]


def test_physics_verifier_routes_by_category():
    v = _dummy_video()
    item = GenerationItem(
        item_id="g1", prompt="p", reference_video_path="/tmp/x",
        domain=Domain.EVERYDAY_PHYSICS,
        physics_category=PhysicsCategory.THERMAL,
        qa_pairs=[],
    )
    verdict = PhysicsVerifier().verify(v, item)
    assert verdict["verifier_type"] == "intractable"

    item.physics_category = PhysicsCategory.RIGID_BODY
    verdict = PhysicsVerifier().verify(v, item)
    assert verdict["verifier_type"] == "analytic"
    assert 0.0 <= verdict["score"] <= 1.0


def test_saturation_detection():
    item_scores = {
        "a": [0.95, 0.92, 0.91, 0.5],
        "b": [0.95, 0.92, 0.7, 0.5],
        "c": [0.9, 0.9, 0.5],
    }
    saturated = detect_saturated_items(item_scores, top_n=3, threshold=0.9)
    assert "a" in saturated
    assert "b" not in saturated
    # "c" has exactly three scores, two >= 0.9 and one below — not saturated.
    assert "c" not in saturated


def test_filter_by_agreement_basic():
    items = [
        QAItem(item_id=f"i{i}", video_path="/tmp/x",
               domain=Domain.EVERYDAY_PHYSICS,
               physics_category=PhysicsCategory.RIGID_BODY,
               question="?", choices=["a", "b", "c", "d"], answer_idx=0,
               requires_temporal=True, language_prior_score=0.0,
               annotator_agreement=0.0) for i in range(4)
    ]
    annotations = {
        "i0": [[0], [0], [0]],
        "i1": [[1], [1], [0]],
        "i2": [[0], [1], [2]],
        "i3": [[0], [0], [0]],
    }
    passed, flagged = filter_by_agreement(items, annotations, threshold=0.5)
    # Total counts add up.
    assert len(passed) + len(flagged) == 4
    # i0 and i3 (full agreement) should pass.
    passed_ids = {i.item_id for i in passed}
    assert "i0" in passed_ids and "i3" in passed_ids
