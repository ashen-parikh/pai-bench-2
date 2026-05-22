"""Analytic physics metrics with no MLLM ceiling.

# PAI-BENCH-2-CHANGE: per slide 1 of the v2 design doc — supplement the
# MLLM-ensemble judge with verifier-grade signals so the overall judge
# can route around the MLLM ceiling for the ~60% of items that admit a
# mechanical check. Each function returns a scalar in [0, 1] (higher =
# more physically plausible) and tolerates missing heavy models via
# documented fallbacks.

Public API:
    optical_flow_smoothness(video) -> float
    depth_stability(video) -> float
    motion_blob_count_stability(video) -> float
    pose_validity(video) -> float | None    # None when no pose detector available

All inputs are (T, H, W, 3) float32 videos in [0, 1].
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_video(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) video; got shape {frames.shape}")
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
    if frames.max() > 1.5:
        frames = frames / 255.0
    return np.clip(frames, 0.0, 1.0)


def _gray_uint8(frames: np.ndarray) -> np.ndarray:
    out = np.empty(frames.shape[:3], dtype=np.uint8)
    for i, f in enumerate(frames):
        out[i] = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return out


def _farneback_flows(gray_u8: np.ndarray) -> np.ndarray:
    """Per-frame Farneback optical flow; returns (T-1, H, W, 2)."""
    flows = []
    for t in range(gray_u8.shape[0] - 1):
        flows.append(cv2.calcOpticalFlowFarneback(
            gray_u8[t], gray_u8[t + 1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        ))
    if not flows:
        return np.zeros((0, *gray_u8.shape[1:], 2), dtype=np.float32)
    return np.stack(flows).astype(np.float32)


# ---------------------------------------------------------------------------
# optical flow smoothness
# ---------------------------------------------------------------------------

def optical_flow_smoothness(frames: np.ndarray) -> float:
    """Temporal smoothness of the optical flow field.

    Computes per-pixel L2 difference between consecutive flow fields and
    squashes the mean to [0, 1] via 1 / (1 + x). Synthetic teleportation,
    morphing, and judder produce large frame-to-frame flow swings and
    therefore lower scores.

    Returns 1.0 when fewer than three frames are present (insufficient to
    measure flow change).
    """
    frames = _ensure_video(frames)
    if frames.shape[0] < 3:
        return 1.0
    gray = _gray_uint8(frames)
    flows = _farneback_flows(gray)
    if flows.shape[0] < 2:
        return 1.0
    delta = np.linalg.norm(np.diff(flows, axis=0), axis=-1)
    mean_delta = float(delta.mean())
    return float(1.0 / (1.0 + mean_delta))


# ---------------------------------------------------------------------------
# depth stability
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_depth_pipe():
    try:
        from transformers import pipeline
        return pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
    except Exception as exc:
        logger.warning("DepthAnythingV2 unavailable (%s); using grayscale proxy", exc)
        return None


def _per_frame_depth(frames: np.ndarray) -> np.ndarray:
    pipe = _load_depth_pipe()
    if pipe is None:
        # Grayscale-luminance proxy: closer-to-camera regions are usually
        # brighter in well-lit scenes. Monotone-only; the metric is robust
        # to a constant shift so this remains a useful ordering signal.
        return _gray_uint8(frames).astype(np.float32) / 255.0
    from PIL import Image
    out = []
    for f in frames:
        img = Image.fromarray((f * 255).astype(np.uint8))
        result = pipe(img)
        d = np.array(result["predicted_depth"], dtype=np.float32)
        if d.max() > 0:
            d = d / d.max()
        # Resize to common shape so depth maps stack into one array.
        h, w = frames.shape[1:3]
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h))
        out.append(d)
    return np.stack(out)


def depth_stability(frames: np.ndarray, n_pairs: int = 256, seed: int = 0) -> float:
    """Pairwise depth-ordering consistency across frames.

    Samples `n_pairs` random pixel pairs and records sign(depth[a] - depth[b])
    in every frame. Returns the fraction of pairs whose sign stays constant
    across the entire clip. A perfectly stable depth ordering scores 1.0;
    randomly inverting depths (e.g. frame-to-frame depth flicker) scores 0.0.

    With <2 frames there's nothing to compare; returns 1.0.
    """
    frames = _ensure_video(frames)
    if frames.shape[0] < 2:
        return 1.0
    depths = _per_frame_depth(frames)
    h, w = depths.shape[1:]
    n_px = h * w
    if n_px < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n_px, size=n_pairs)
    b = rng.integers(0, n_px, size=n_pairs)
    flat = depths.reshape(depths.shape[0], -1)
    signs = np.sign(flat[:, a] - flat[:, b])      # (T, n_pairs)
    # A pair is stable if its sign is the same in every frame.
    first = signs[0]
    stable = np.all(signs == first[None, :], axis=0)
    return float(stable.mean())


# ---------------------------------------------------------------------------
# motion-blob count stability (lightweight object-tracking proxy)
# ---------------------------------------------------------------------------

def motion_blob_count_stability(frames: np.ndarray, min_area: int = 32) -> float:
    """How stable is the count of large motion blobs across the clip?

    For each adjacent pair, computes the absolute difference, thresholds it at
    mean + std, and counts connected components above `min_area` pixels.
    Returns a score derived from the standard deviation of those counts.
    Phantom-object appearance/disappearance increases variance and lowers the
    score; a single tracked object yielding a stable count scores near 1.0.
    """
    frames = _ensure_video(frames)
    if frames.shape[0] < 2:
        return 1.0
    counts: list[int] = []
    gray = _gray_uint8(frames).astype(np.float32) / 255.0
    for t in range(gray.shape[0] - 1):
        diff = np.abs(gray[t + 1] - gray[t])
        if diff.max() < 1e-3:
            counts.append(0)
            continue
        thresh = float(diff.mean() + diff.std())
        mask = (diff > thresh).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # stats[0] is the background; count components above min_area.
        big = int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= min_area))
        counts.append(big)
    if not counts:
        return 1.0
    return float(1.0 / (1.0 + np.std(counts)))


# ---------------------------------------------------------------------------
# pose validity (stub)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_pose_detector():
    """Try to load a lightweight pose detector. Returns None if unavailable.

    We deliberately avoid hard-depending on MediaPipe / HRNet because they
    drag in significant build complexity on macOS/M1. The function below
    returns None when no detector loads — callers should treat that as
    "skipped", not "failed".
    """
    try:
        import mediapipe as mp
        return mp.solutions.pose.Pose(static_image_mode=False, model_complexity=0)
    except Exception as exc:
        logger.info("pose detector unavailable (%s); pose_validity will return None", exc)
        return None


def pose_validity(frames: np.ndarray) -> float | None:
    """Bone-length consistency for detected human/robot poses.

    When a pose detector is available, extracts keypoints per frame and
    measures coefficient of variation of inter-joint distances; very high
    CV across frames flags non-rigid skeleton deformation. Returns a score
    in [0, 1].

    Returns None when no pose detector is loadable or when no pose is
    detected in any frame. None is a meaningful signal: the caller should
    drop this check rather than treat it as zero.
    """
    frames = _ensure_video(frames)
    detector = _load_pose_detector()
    if detector is None:
        return None
    # Adjacent connected keypoints (MediaPipe pose convention).
    edges = [(11, 13), (13, 15), (12, 14), (14, 16),    # arms
             (23, 25), (25, 27), (24, 26), (26, 28),    # legs
             (11, 12), (23, 24), (11, 23), (12, 24)]    # torso

    lengths: dict[tuple[int, int], list[float]] = {e: [] for e in edges}
    detected_any = False
    for f in frames:
        result = detector.process((f * 255).astype(np.uint8))
        if not getattr(result, "pose_landmarks", None):
            continue
        detected_any = True
        lm = result.pose_landmarks.landmark
        for a, b in edges:
            dx = lm[a].x - lm[b].x
            dy = lm[a].y - lm[b].y
            lengths[(a, b)].append(float(np.sqrt(dx * dx + dy * dy)))
    if not detected_any:
        return None

    # Coefficient of variation per edge, averaged. CoV in [0, ~1+] for
    # well-detected stable skeletons; squash to [0,1].
    cvs = []
    for vals in lengths.values():
        if len(vals) < 2:
            continue
        mean = float(np.mean(vals))
        if mean < 1e-6:
            continue
        cvs.append(float(np.std(vals) / mean))
    if not cvs:
        return None
    return float(1.0 / (1.0 + float(np.mean(cvs))))
