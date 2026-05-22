"""Anthropic Claude MLLM judge.

Mirror of `MLLMJudge` for the Anthropic SDK. Used as an ensemble member when
the operator wants a cross-vendor judge panel (mixing OpenAI + Anthropic
reduces the chance the ensemble's median is dominated by a single vendor's
training-data quirks).

# PAI-BENCH-2-CHANGE: complements MLLMJudge (OpenAI) so EnsembleJudge can
# be assembled across vendors. Same BaseJudge contract; same fallback
# semantics (score=None on call failure — never a fake mid-range number).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import warnings
from typing import Any

import numpy as np

from pai_bench.data.schema import GenerationItem
from pai_bench.judge.base import BaseJudge

logger = logging.getLogger(__name__)

_DEPRECATION_NOTE = (
    "AnthropicMLLMJudge is used here only because the analytic verifier marked "
    "this physics category as intractable. Outputs carry uncertainty='high' "
    "and should not be the primary signal in the leaderboard."
)


class AnthropicMLLMJudge(BaseJudge):
    judge_type = "anthropic_mllm"

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
        max_tokens: int = 512,
    ):
        warnings.warn(_DEPRECATION_NOTE, RuntimeWarning, stacklevel=2)
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._client = None

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key) if self.api_key else None
        except Exception as exc:
            logger.warning("anthropic package unavailable (%s)", exc)
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
            logger.warning("No Anthropic client; returning unscored verdict for %s", item.item_id)
            return {
                "score": None,                              # honest: failed to score
                "judge_type": f"{self.judge_type}:{self.model}",
                "uncertainty": "high",
                "violations": [],
                "rationale": "no_client",
            }

        frames = self._sample_frames(video)
        from PIL import Image
        encoded: list[str] = []
        for f in frames:
            img = Image.fromarray((f * 255).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            encoded.append(base64.b64encode(buf.getvalue()).decode())

        question = (
            f"Item: {item.item_id}. Domain: {item.domain}. "
            f"Physics: {item.physics_category}.\n"
            f"Prompt that produced this video: {item.prompt}\n"
            f"Expected physics behavior: {item.expected_physics}\n"
            "Rate physical plausibility on a 0-1 scale. Reply with the number on "
            "the first line, then list any specific physics violations."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": question}]
        for b in encoded:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b},
            })

        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        except Exception as exc:
            logger.warning("Anthropic judge call failed (%s)", exc)
            return {
                "score": None,                              # honest: failed to score
                "judge_type": f"{self.judge_type}:{self.model}",
                "uncertainty": "high",
                "violations": [],
                "rationale": f"call_failed_{exc!s}",
            }

        m = re.search(r"([0-1](?:\.\d+)?)", text)
        score = float(m.group(1)) if m else 0.5
        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "judge_type": f"{self.judge_type}:{self.model}",
            "uncertainty": "medium",
            "violations": [],
            "rationale": text,
        }
