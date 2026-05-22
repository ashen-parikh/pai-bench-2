"""Language-prior auditor.

For each QA item, runs a text-only LLM (no video) n_trials times and records
the fraction it gets correct. Items above the threshold are filtered: they
reward language bias rather than video understanding.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable

from tqdm import tqdm

from pai_bench.data.schema import QAItem

logger = logging.getLogger(__name__)


class LanguagePriorAuditor:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        threshold: float = 0.60,
        api_key: str | None = None,
        client_fn: Callable | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # client_fn allows tests to inject a deterministic responder.
        self._client_fn = client_fn

    def _client(self):
        if self._client_fn is not None:
            return self._client_fn
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key) if self.api_key else None
        except Exception as exc:
            logger.warning("openai unavailable (%s); auditor will return 0.0", exc)
            return None

        def _fn(question: str, choices: list[str]) -> int:
            if client is None:
                return -1
            prompt = (
                "You will be asked a question with 4 choices. You CANNOT see the video. "
                "Pick the most likely answer based on the question text alone. "
                "Respond with a single integer 0-3.\n\n"
                f"Question: {question}\n"
                + "\n".join(f"{i}: {c}" for i, c in enumerate(choices))
            )
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"[0-3]", text)
            return int(m.group()) if m else -1

        return _fn

    def audit_item(self, item: QAItem, n_trials: int = 5) -> float:
        fn = self._client()
        if fn is None:
            return 0.0
        correct = 0
        for _ in range(n_trials):
            pred = fn(item.question, item.choices)
            if pred == item.answer_idx:
                correct += 1
        return correct / n_trials

    def audit_batch(self, items: list[QAItem], n_trials: int = 5) -> list[QAItem]:
        rejected_by_domain: dict[str, int] = {}
        passed: list[QAItem] = []
        for item in tqdm(items, desc="lang-prior audit"):
            score = self.audit_item(item, n_trials)
            item.language_prior_score = score
            if score < self.threshold:
                passed.append(item)
            else:
                dom = item.domain if isinstance(item.domain, str) else item.domain.value
                rejected_by_domain[dom] = rejected_by_domain.get(dom, 0) + 1
        for dom, n in rejected_by_domain.items():
            logger.info("rejected %d items in domain %s (lang prior >= %.2f)",
                        n, dom, self.threshold)
        return passed
