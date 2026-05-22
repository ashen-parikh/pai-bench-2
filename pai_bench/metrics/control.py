"""Control fidelity metrics for track C.

Each metric compares a generated video to a reference and returns a scalar in
[0, 1] where 1 = perfect adherence to the control signal.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


def _to_uint8_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    return frame


def _align(generated: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim to the shorter length so frame indices align."""
    T = min(generated.shape[0], reference.shape[0])
    return generated[:T], reference[:T]


def blur_ssim(generated: np.ndarray, reference: np.ndarray,
              sigma: float = 2.0) -> float:
    """Gaussian-blurred SSIM, mean over frames. Used for ControlNet-style metrics."""
    g, r = _align(generated, reference)
    scores = []
    k = max(3, int(2 * sigma * 3) | 1)        # odd kernel size
    for fg, fr in zip(g, r):
        fg = cv2.GaussianBlur(fg, (k, k), sigma)
        fr = cv2.GaussianBlur(fr, (k, k), sigma)
        if fg.ndim == 3:
            scores.append(ssim(fg, fr, channel_axis=-1, data_range=1.0))
        else:
            scores.append(ssim(fg, fr, data_range=1.0))
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _canny_edges(frame: np.ndarray) -> np.ndarray:
    img = _to_uint8_gray(frame)
    v = float(np.median(img))
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    return cv2.Canny(img, lower, upper) > 0


def edge_f1(generated: np.ndarray, reference: np.ndarray) -> float:
    """F1 over Canny edge maps, mean over frames."""
    g, r = _align(generated, reference)
    f1s = []
    for fg, fr in zip(g, r):
        eg = _canny_edges(fg)
        er = _canny_edges(fr)
        tp = float(np.logical_and(eg, er).sum())
        fp = float(np.logical_and(eg, ~er).sum())
        fn = float(np.logical_and(~eg, er).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.clip(np.mean(f1s), 0.0, 1.0))


@lru_cache(maxsize=1)
def _load_depth():
    try:
        from transformers import pipeline
        return pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
    except Exception as exc:
        logger.warning("DepthAnythingV2 unavailable (%s); using grayscale proxy", exc)
        return None


def _depth_map(frame: np.ndarray, pipe) -> np.ndarray:
    if pipe is None:
        return cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    from PIL import Image
    img = Image.fromarray((frame * 255).astype(np.uint8))
    out = pipe(img)
    d = np.array(out["predicted_depth"], dtype=np.float32)
    if d.max() > 0:
        d = d / d.max()
    return d


def depth_si_rmse(generated: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant depth RMSE inverted to [0,1] (1 = identical depth)."""
    g, r = _align(generated, reference)
    pipe = _load_depth()
    losses = []
    for fg, fr in zip(g, r):
        dp = _depth_map(fg, pipe) + 1e-3
        dt = _depth_map(fr, pipe) + 1e-3
        if dp.shape != dt.shape:
            dp = cv2.resize(dp, (dt.shape[1], dt.shape[0]))
        d = np.log(dp) - np.log(dt)
        si_rmse = float(np.sqrt(np.mean(d ** 2) - np.mean(d) ** 2))
        losses.append(si_rmse)
    # Map RMSE in log space to [0,1] via 1/(1+x).
    return float(1.0 / (1.0 + np.mean(losses)))


@lru_cache(maxsize=1)
def _load_segformer():
    try:
        from transformers import pipeline
        return pipeline(
            "image-segmentation",
            model="nvidia/segformer-b0-finetuned-ade-512-512",
        )
    except Exception as exc:
        logger.warning("SegFormer unavailable (%s); using thresholding proxy", exc)
        return None


def _seg_mask(frame: np.ndarray, pipe) -> np.ndarray:
    if pipe is None:
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return (gray > 128).astype(np.int32)
    from PIL import Image
    img = Image.fromarray((frame * 255).astype(np.uint8))
    segs = pipe(img)
    # Combine per-class binary masks into a single integer-labelled map.
    h, w = (frame.shape[0], frame.shape[1])
    out = np.zeros((h, w), dtype=np.int32)
    for i, seg in enumerate(segs):
        m = np.array(seg["mask"])
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out = np.where(m > 0, i + 1, out)
    return out


def mask_miou(generated: np.ndarray, reference: np.ndarray) -> float:
    """Mean IoU between segmentation maps, averaged over frames."""
    g, r = _align(generated, reference)
    pipe = _load_segformer()
    ious = []
    for fg, fr in zip(g, r):
        mg = _seg_mask(fg, pipe)
        mr = _seg_mask(fr, pipe)
        labels = np.unique(np.concatenate([np.unique(mg), np.unique(mr)]))
        per_class = []
        for lbl in labels:
            inter = np.logical_and(mg == lbl, mr == lbl).sum()
            union = np.logical_or(mg == lbl, mr == lbl).sum()
            if union > 0:
                per_class.append(inter / union)
        ious.append(float(np.mean(per_class)) if per_class else 0.0)
    return float(np.clip(np.mean(ious), 0.0, 1.0))


def robustness_score(
    generated_videos: dict[int, np.ndarray],
    reference: np.ndarray,
    primary_metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """
    PAI-BENCH-2-CHANGE: new metric. Measure graceful degradation.

    Given a mapping of degradation_level (0..3) -> generated video, compute
    primary_metric_fn at each level and report the normalised AUC of the
    fidelity-vs-degradation curve. Higher = more robust.
    """
    if not generated_videos:
        return 0.0
    levels = sorted(generated_videos.keys())
    fidelities = [
        float(primary_metric_fn(generated_videos[lvl], reference)) for lvl in levels
    ]
    if len(levels) == 1:
        return float(np.clip(fidelities[0], 0.0, 1.0))
    # Normalised AUC: trapezoid divided by (range_of_levels * 1).
    auc = float(np.trapz(fidelities, x=levels))
    span = levels[-1] - levels[0]
    return float(np.clip(auc / span if span > 0 else fidelities[0], 0.0, 1.0))
