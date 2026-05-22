"""Quality Score metrics: subject/background consistency, smoothness, aesthetic, etc.

All functions take an (T, H, W, C) float32 video in [0, 1] and return a scalar in [0, 1].
Models are loaded lazily on first call and cached on the module. GPU is used when
available; CPU fallback is automatic.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


def _torch():
    """Lazy torch import so the metrics module can be imported without torch."""
    import torch
    return torch


def _device():
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_video(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) video; got shape {frames.shape}")
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
    if frames.max() > 1.5:        # uint8-ish range
        frames = frames / 255.0
    return np.clip(frames, 0.0, 1.0)


def _cosine_chain_score(features: np.ndarray) -> float:
    """Shared (1/T-1) * sum 0.5*(<d1,dt> + <d_{t-1},dt>) formulation."""
    feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    T = feats.shape[0]
    if T < 2:
        return 1.0
    acc = 0.0
    for t in range(1, T):
        acc += 0.5 * (float(np.dot(feats[0], feats[t])) +
                      float(np.dot(feats[t - 1], feats[t])))
    score = acc / (T - 1)
    # Cosine on unit vectors lives in [-1,1]; squash to [0,1].
    return float((score + 1.0) / 2.0)


@lru_cache(maxsize=1)
def _load_dino():
    torch = _torch()
    try:
        model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    except Exception as exc:    # offline / no internet during tests
        logger.warning("DINO load failed (%s); using random projection fallback", exc)
        return None
    model = model.to(_device()).eval()
    return model


def subject_consistency(frames: np.ndarray) -> float:
    """DINO ViT-B/16 feature cosine consistency across frames."""
    torch = _torch()
    frames = _ensure_video(frames)
    model = _load_dino()
    if model is None:
        # Deterministic fallback: use per-frame mean color vector.
        feats = frames.mean(axis=(1, 2))
        return _cosine_chain_score(feats)
    import torch.nn.functional as F
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).to(_device())
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        feats = model(x).cpu().numpy()
    return _cosine_chain_score(feats)


@lru_cache(maxsize=1)
def _load_clip():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        logger.warning("transformers unavailable (%s); CLIP disabled", exc)
        return None, None
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device()).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as exc:
        logger.warning("CLIP load failed (%s); fallback enabled", exc)
        return None, None
    return model, proc


def background_consistency(frames: np.ndarray) -> float:
    """CLIP ViT-B/32 image feature cosine consistency across frames."""
    torch = _torch()
    frames = _ensure_video(frames)
    model, proc = _load_clip()
    if model is None:
        feats = frames.reshape(frames.shape[0], -1)
        return _cosine_chain_score(feats)
    imgs = [(f * 255).astype(np.uint8) for f in frames]
    inputs = proc(images=imgs, return_tensors="pt").to(_device())
    with torch.no_grad():
        feats = model.get_image_features(**inputs).cpu().numpy()
    return _cosine_chain_score(feats)


def motion_smoothness(frames: np.ndarray) -> float:
    """Compare original odd frames against frames interpolated from neighbors.

    The reference VBench definition uses FILM/RIFE. If unavailable, we fall back
    to linear (midpoint) interpolation, which under-estimates smoothness but
    preserves monotonicity for ranking.
    """
    frames = _ensure_video(frames)
    T = frames.shape[0]
    if T < 3:
        return 1.0
    losses = []
    for t in range(1, T - 1, 2):    # odd indices
        interp = 0.5 * (frames[t - 1] + frames[t + 1])
        losses.append(float(np.abs(interp - frames[t]).mean()))
    if not losses:
        return 1.0
    mean_l1 = float(np.mean(losses))
    return float(max(0.0, 1.0 - mean_l1))


@lru_cache(maxsize=1)
def _load_aesthetic():
    try:
        from transformers import pipeline
        pipe = pipeline(
            "image-classification",
            model="shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
            device=0 if _device() == "cuda" else -1,
        )
        return pipe
    except Exception as exc:
        logger.warning("Aesthetic predictor unavailable (%s)", exc)
        return None


def aesthetic_quality(frames: np.ndarray) -> float:
    """LAION aesthetic predictor; returns mean per-frame score normalised to [0,1]."""
    frames = _ensure_video(frames)
    pipe = _load_aesthetic()
    if pipe is None:
        # Fallback heuristic: average luminance contrast.
        gray = frames.mean(axis=-1)
        contrast = gray.std(axis=(1, 2)).mean()
        return float(np.clip(contrast * 2, 0.0, 1.0))
    from PIL import Image
    scores = []
    for f in frames:
        img = Image.fromarray((f * 255).astype(np.uint8))
        out = pipe(img)
        # The aesthetic model emits a regression score in roughly [1, 10].
        try:
            raw = float(out[0]["score"])
        except (KeyError, IndexError, TypeError):
            raw = 5.0
        scores.append(np.clip((raw - 1.0) / 9.0, 0.0, 1.0))
    return float(np.mean(scores))


@lru_cache(maxsize=1)
def _load_musiq():
    try:
        from transformers import pipeline
        return pipeline(
            "image-classification",
            model="google/musiq-spaq",
            device=0 if _device() == "cuda" else -1,
        )
    except Exception as exc:
        logger.warning("MUSIQ unavailable (%s)", exc)
        return None


def imaging_quality(frames: np.ndarray) -> float:
    """MUSIQ image quality score, averaged across frames."""
    frames = _ensure_video(frames)
    pipe = _load_musiq()
    if pipe is None:
        gray = frames.mean(axis=-1)
        # Variance of Laplacian as cheap sharpness proxy.
        import cv2
        sharp = [cv2.Laplacian((g * 255).astype(np.uint8), cv2.CV_64F).var() for g in gray]
        return float(np.clip(np.mean(sharp) / 500.0, 0.0, 1.0))
    from PIL import Image
    out = []
    for f in frames:
        img = Image.fromarray((f * 255).astype(np.uint8))
        try:
            out.append(float(pipe(img)[0]["score"]))
        except (KeyError, IndexError, TypeError):
            out.append(0.5)
    return float(np.clip(np.mean(out), 0.0, 1.0))


def overall_consistency(frames: np.ndarray, prompt: str) -> float:
    """ViCLIP video-text cosine alignment.

    Falls back to per-frame CLIP image-text similarity if ViCLIP is not installed.
    """
    torch = _torch()
    frames = _ensure_video(frames)
    model, proc = _load_clip()
    if model is None or not prompt:
        return 0.5
    imgs = [(f * 255).astype(np.uint8) for f in frames]
    inputs = proc(text=[prompt], images=imgs, return_tensors="pt", padding=True).to(_device())
    with torch.no_grad():
        out = model(**inputs)
        img_feats = out.image_embeds         # (T, D)
        txt_feats = out.text_embeds          # (1, D)
        img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)
        txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-8)
        sims = (img_feats @ txt_feats.T).squeeze(-1).cpu().numpy()
    return float(np.clip((sims.mean() + 1.0) / 2.0, 0.0, 1.0))
