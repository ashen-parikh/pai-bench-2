"""MLLM-as-Judge.

# PAI-BENCH-2-CHANGE: this judge remains available as a fallback for physics
# categories the analytic verifier cannot handle (thermal, deformable, EM).
# It now emits an explicit "uncertainty": "high" flag and is never the
# default choice for tractable categories — see hybrid_judge.HybridJudge.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import numpy as np

from pai_bench.data.schema import GenerationItem
from pai_bench.judge.base import BaseJudge

logger = logging.getLogger(__name__)

_DEPRECATION_NOTE = (
    "MLLMJudge is used here only because the analytic verifier marked this "
    "physics category as intractable. Outputs carry uncertainty='high' and "
    "should not be the primary signal in the leaderboard."
)


class MLLMJudge(BaseJudge):
    judge_type = "mllm"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        warnings.warn(_DEPRECATION_NOTE, RuntimeWarning, stacklevel=2)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key) if self.api_key else None
        except Exception as exc:
            logger.warning("openai package unavailable (%s)", exc)
            self._client = None
        return self._client

    def _sample_frames(self, video: np.ndarray, n: int = 8) -> list[np.ndarray]:
        if video.shape[0] <= n:
            return list(video)
        idx = np.linspace(0, video.shape[0] - 1, n).astype(int)
        return [video[i] for i in idx]

    def score(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        client = self._client_lazy()
        if client is None:
            logger.warning("No OpenAI client; returning unscored verdict for %s", item.item_id)
            # PAI-BENCH-2-CHANGE: return score=None, not 0.5. Three judges that
            # all fail in the same way (e.g. same API quota error) used to look
            # like consensus to EnsembleJudge; the None signal is filtered out
            # of the aggregate so disagreement reporting stays honest.
            return {
                "score": None,
                "judge_type": self.judge_type,
                "uncertainty": "high",
                "violations": [],
                "rationale": "no_client",
            }

        frames = self._sample_frames(video)
        # Encode frames as base64 PNGs.
        import base64
        import io
        from PIL import Image

        encoded = []
        for f in frames:
            img = Image.fromarray((f * 255).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            encoded.append(base64.b64encode(buf.getvalue()).decode())

        question = (
            f"Item: {item.item_id}. Domain: {item.domain}. Physics: {item.physics_category}.\n"
            f"Prompt that produced this video: {item.prompt}\n"
            f"Expected physics behavior: {item.expected_physics}\n"
            f"Rate physical plausibility on a 0-1 scale and list specific violations."
        )
        content = [{"type": "text", "text": question}]
        for b in encoded:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b}"},
            })
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("MLLM judge call failed (%s)", exc)
            return {
                "score": None,                  # honest: failed to score; ensemble filters it out
                "judge_type": self.judge_type,
                "uncertainty": "high",
                "violations": [],
                "rationale": f"call_failed_{exc!s}",
            }
        # Extract a leading number from the response if present.
        import re
        m = re.search(r"([0-1](?:\.\d+)?)", text)
        score = float(m.group(1)) if m else 0.5
        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "judge_type": self.judge_type,
            "uncertainty": "high",
            "violations": [],
            "rationale": text,
        }
