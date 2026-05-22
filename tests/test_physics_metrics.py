"""Analytic physics metrics tests using synthetic videos.

These exercise the fallback codepaths (no heavy model downloads required)
and verify each metric returns a value in [0, 1] with the expected
qualitative behavior on clean vs corrupted motion.
"""

from __future__ import annotations

import numpy as np
import pytest

from pai_bench.metrics import physics_metrics as pm


def _moving_square(t: int = 12, h: int = 64, w: int = 96, jitter: float = 0.0,
                   seed: int = 0) -> np.ndarray:
    """Square translating rightward at constant speed, optionally with frame jitter."""
    rng = np.random.default_rng(seed)
    frames = np.full((t, h, w, 3), 0.05, dtype=np.float32)
    x = 4.0
    for i in range(t):
        xi = int(x + (rng.normal() * jitter if jitter > 0 else 0))
        xi = max(0, min(w - 12, xi))
        frames[i, h // 2 - 6:h // 2 + 6, xi:xi + 12, :] = 1.0
        x += 4.0
    return frames


def _teleporting_square(t: int = 12, h: int = 64, w: int = 96,
                        seed: int = 0) -> np.ndarray:
    """Square that randomly jumps around — no coherent motion."""
    rng = np.random.default_rng(seed)
    frames = np.full((t, h, w, 3), 0.05, dtype=np.float32)
    for i in range(t):
        x = int(rng.integers(0, w - 12))
        y = int(rng.integers(0, h - 12))
        frames[i, y:y + 12, x:x + 12, :] = 1.0
    return frames


def _depth_flicker(t: int = 8, h: int = 48, w: int = 48, seed: int = 0) -> np.ndarray:
    """Frame brightness flips frame-to-frame — depth proxy will flicker."""
    rng = np.random.default_rng(seed)
    frames = np.empty((t, h, w, 3), dtype=np.float32)
    for i in range(t):
        # Alternate dark/light halves.
        frames[i, :, :w // 2, :] = 0.9 if i % 2 == 0 else 0.1
        frames[i, :, w // 2:, :] = 0.1 if i % 2 == 0 else 0.9
    return frames


# ---------------------------------------------------------------------------
# optical_flow_smoothness
# ---------------------------------------------------------------------------

def test_flow_smoothness_in_unit_range():
    v = _moving_square()
    assert 0.0 <= pm.optical_flow_smoothness(v) <= 1.0


def test_flow_smoothness_higher_for_constant_motion_than_teleport():
    clean = pm.optical_flow_smoothness(_moving_square())
    chaos = pm.optical_flow_smoothness(_teleporting_square())
    assert clean > chaos


def test_flow_smoothness_too_few_frames_returns_one():
    v = _moving_square(t=2)
    assert pm.optical_flow_smoothness(v) == 1.0


# ---------------------------------------------------------------------------
# depth_stability
# ---------------------------------------------------------------------------

def test_depth_stability_in_unit_range():
    v = _moving_square()
    score = pm.depth_stability(v)
    assert 0.0 <= score <= 1.0


def test_depth_stability_high_for_static_depth_low_for_flicker():
    static = pm.depth_stability(_moving_square())
    flick = pm.depth_stability(_depth_flicker())
    # Translating-square scene preserves most depth orderings; flicker scene
    # inverts them every frame.
    assert static > flick


# ---------------------------------------------------------------------------
# motion_blob_count_stability
# ---------------------------------------------------------------------------

def test_blob_count_stability_in_unit_range():
    v = _moving_square()
    assert 0.0 <= pm.motion_blob_count_stability(v) <= 1.0


def test_blob_count_stability_higher_for_single_object():
    single = pm.motion_blob_count_stability(_moving_square())
    chaos = pm.motion_blob_count_stability(_teleporting_square())
    assert single >= chaos


# ---------------------------------------------------------------------------
# pose_validity (stub)
# ---------------------------------------------------------------------------

def test_pose_validity_skips_when_no_detector():
    """We don't depend on mediapipe; without it, pose_validity should return None."""
    v = _moving_square()
    result = pm.pose_validity(v)
    # Either None (skipped) or a float in [0,1] if a detector loads anyway.
    assert result is None or 0.0 <= result <= 1.0
