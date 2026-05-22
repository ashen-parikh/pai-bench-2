"""Diversity metric used by track C for variant-prompt generations."""

from __future__ import annotations

import logging
from functools import lru_cache
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _lpips_model():
    try:
        import lpips
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = lpips.LPIPS(net="alex").to(device).eval()
        return model, device
    except Exception as exc:
        logger.warning("LPIPS unavailable (%s); using L2 fallback", exc)
        return None, None


def _to_tensor(video: np.ndarray):
    import torch
    x = torch.from_numpy(video.astype(np.float32)).permute(0, 3, 1, 2)
    x = x * 2.0 - 1.0          # LPIPS expects [-1,1]
    return x


def _pairwise_distance(va: np.ndarray, vb: np.ndarray) -> float:
    model, device = _lpips_model()
    T = min(va.shape[0], vb.shape[0])
    if model is None:
        return float(np.mean(np.abs(va[:T] - vb[:T])))
    import torch
    xa = _to_tensor(va[:T]).to(device)
    xb = _to_tensor(vb[:T]).to(device)
    with torch.no_grad():
        d = model(xa, xb).cpu().numpy().reshape(-1)
    return float(d.mean())


def generation_diversity(videos: list[np.ndarray]) -> float:
    """Mean pairwise LPIPS over a list of videos."""
    if len(videos) < 2:
        return 0.0
    dists = [_pairwise_distance(a, b) for a, b in combinations(videos, 2)]
    # LPIPS in [0, ~1+] range; squash with min(1, x).
    return float(np.clip(np.mean(dists), 0.0, 1.0))
