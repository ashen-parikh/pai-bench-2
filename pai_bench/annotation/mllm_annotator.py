"""MLLM-assisted first-pass QA generation.

Given a video clip + ontology category, asks an MLLM to draft (question,
choices, answer_idx) candidates. Outputs are NOT considered ground truth —
they enter human_review for adjudication before any item lands in the
benchmark.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Iterable

from pai_bench.data.schema import Domain, PhysicsCategory, QAItem

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """\
You are drafting multiple-choice questions for a physical-reasoning benchmark.
Watch the video and produce ONE question that requires temporal reasoning to
answer correctly (a text-only model should NOT be able to guess the answer
above 60% accuracy from the question alone).

Domain: {domain}
Physics category: {physics_category}
Ontology branch: {ontology_branch}

Return strict JSON:
{{
  "question": "...",
  "choices": ["...", "...", "...", "..."],
  "answer_idx": 0,
  "requires_temporal": true,
  "tags": ["..."]
}}
"""


class MLLMAnnotator:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _client_lazy(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key) if self.api_key else None
        except Exception as exc:
            logger.warning("openai unavailable (%s); annotator will return [].", exc)
        return self._client

    def draft(
        self,
        video_path: Path,
        domain: Domain,
        physics_category: PhysicsCategory,
        ontology_branch: str,
        n_candidates: int = 3,
    ) -> list[QAItem]:
        client = self._client_lazy()
        if client is None:
            return []
        prompt = PROMPT_TEMPLATE.format(
            domain=domain.value if isinstance(domain, Domain) else domain,
            physics_category=physics_category.value if isinstance(physics_category, PhysicsCategory) else physics_category,
            ontology_branch=ontology_branch,
        )
        out: list[QAItem] = []
        for _ in range(n_candidates):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                )
                text = resp.choices[0].message.content or ""
            except Exception as exc:
                logger.warning("annotator call failed: %s", exc)
                continue
            payload = self._extract_json(text)
            if payload is None:
                continue
            try:
                item = QAItem(
                    item_id=str(uuid.uuid4()),
                    video_path=str(video_path),
                    domain=domain,
                    physics_category=physics_category,
                    question=payload["question"],
                    choices=payload["choices"],
                    answer_idx=int(payload["answer_idx"]),
                    requires_temporal=bool(payload.get("requires_temporal", True)),
                    # Placeholders; filled in by language_prior and itr_filter.
                    language_prior_score=0.0,
                    annotator_agreement=0.0,
                    tags=list(payload.get("tags", [])),
                )
                out.append(item)
            except Exception as exc:
                logger.warning("invalid candidate from MLLM (%s): %s", exc, payload)
        return out

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
