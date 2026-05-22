"""Physics verifier tests on synthetic motion videos.

We synthesise tiny videos with known dynamics (a falling square, a sliding
square) and check the verifier's output structure plus the high-level pass
of obvious cases.
"""

from __future__ import annotations

import numpy as np

from pai_bench.data.schema import Domain, GenerationItem, PhysicsCategory
from pai_bench.metrics.physics_verifier import PhysicsVerifier


def _falling_square(t: int = 16, h: int = 96, w: int = 96) -> np.ndarray:
    """A square that translates downward with constant acceleration."""
    frames = np.full((t, h, w, 3), 0.05, dtype=np.float32)
    for i in range(t):
        y = int(8 + i * i * 0.25)
        if y + 12 > h:
            break
        frames[i, y:y + 12, w // 2 - 6:w // 2 + 6, :] = 1.0
    return frames


def _sliding_square(t: int = 16, h: int = 64, w: int = 96) -> np.ndarray:
    """A square that translates rightward with decreasing speed (friction)."""
    frames = np.full((t, h, w, 3), 0.05, dtype=np.float32)
    x = 4.0
    v = 6.0
    for i in range(t):
        if x + 12 > w:
            break
        xi = int(x)
        frames[i, h // 2 - 6:h // 2 + 6, xi:xi + 12, :] = 1.0
        x += v
        v *= 0.85
    return frames


def _item(category: PhysicsCategory) -> GenerationItem:
    return GenerationItem(
        item_id="syn", prompt="p", reference_video_path="/tmp/x",
        domain=Domain.EVERYDAY_PHYSICS,
        physics_category=category,
        qa_pairs=[],
        expected_physics={"gravity_direction": [0.0, 1.0], "expected_collisions": 0},
    )


def test_rigid_body_falling_square_aligned_with_gravity():
    v = _falling_square()
    verdict = PhysicsVerifier().verify(v, _item(PhysicsCategory.RIGID_BODY))
    assert verdict["verifier_type"] == "analytic"
    assert verdict["score"] >= 0.5


def test_contact_sliding_square_decelerates():
    v = _sliding_square()
    verdict = PhysicsVerifier().verify(v, _item(PhysicsCategory.CONTACT))
    assert verdict["verifier_type"] == "analytic"
    assert verdict["score"] >= 0.5
    assert "friction_violation_speeding_up" not in verdict["violations"]


def test_intractable_category_returns_marker():
    v = _falling_square()
    verdict = PhysicsVerifier().verify(v, _item(PhysicsCategory.THERMAL))
    assert verdict["verifier_type"] == "intractable"
    assert verdict["score"] is None


def test_supplementary_scores_attached_to_verdict():
    """Rigid-body verdict should include the new analytic physics_metrics scores."""
    v = _falling_square()
    verdict = PhysicsVerifier().verify(v, _item(PhysicsCategory.RIGID_BODY))
    assert "supplementary_scores" in verdict
    assert "flow_smoothness" in verdict["supplementary_scores"]
    assert "depth_stability" in verdict["supplementary_scores"]
    assert "blob_count_stability" in verdict["supplementary_scores"]
    # pose_validity may be None if no detector is installed; that's fine.
    assert "pose_validity" in verdict["supplementary_scores"]
    # Each non-None score is a unit-interval scalar.
    for name, val in verdict["supplementary_scores"].items():
        if val is not None:
            assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"


def test_checks_dict_records_per_check_results():
    v = _falling_square()
    verdict = PhysicsVerifier().verify(v, _item(PhysicsCategory.RIGID_BODY))
    assert "checks" in verdict
    # Original three rigid-body checks must be present.
    assert {"gravity_alignment", "collision_count", "no_interpenetration"} <= verdict["checks"].keys()
    # Supplementary checks (those with non-None scores) must also be present.
    assert "flow_smoothness" in verdict["checks"]
