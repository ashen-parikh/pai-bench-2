"""Counterfactual-track scoring helpers.

# PAI-BENCH-2-CHANGE: entirely new metric family. Tests whether a model has
# a causal world model rather than just a static scene-description ability.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import numpy as np

from pai_bench.data.schema import CounterfactualItem

logger = logging.getLogger(__name__)

DIRECTION_KEYWORDS = {
    "faster": "faster", "slower": "slower",
    "higher": "higher", "lower": "lower",
    "more": "more", "less": "less",
    "left": "left", "right": "right",
    "up": "up", "down": "down",
    "longer": "longer", "shorter": "shorter",
    "earlier": "earlier", "later": "later",
}

MAGNITUDE_BINS = [
    "much_less", "slightly_less", "same", "slightly_more", "much_more"
]
_MAGNITUDE_PHRASES = {
    "much_less": [r"much\s+less", r"far\s+less", r"significantly\s+less"],
    "slightly_less": [r"slightly\s+less", r"a\s+bit\s+less", r"a\s+little\s+less"],
    "same": [r"\bsame\b", r"unchanged", r"no\s+change"],
    "slightly_more": [r"slightly\s+more", r"a\s+bit\s+more", r"a\s+little\s+more"],
    "much_more": [r"much\s+more", r"far\s+more", r"significantly\s+more"],
}


@lru_cache(maxsize=1)
def _nli():
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            top_k=None,
        )
    except Exception as exc:
        logger.warning("NLI model unavailable (%s); using string-overlap fallback", exc)
        return None


def _entailment_prob(premise: str, hypothesis: str) -> float:
    pipe = _nli()
    if pipe is None:
        # Crude token overlap fallback.
        a = set(premise.lower().split())
        b = set(hypothesis.lower().split())
        if not b:
            return 0.0
        return len(a & b) / len(b)
    text = f"{premise} [SEP] {hypothesis}"
    out = pipe(text)
    # pipeline returns list of {label, score} per example.
    if isinstance(out, list) and out and isinstance(out[0], list):
        out = out[0]
    for entry in out:
        if entry["label"].lower().startswith("entail"):
            return float(entry["score"])
    return 0.0


_OPPOSITES = {
    "faster": "slower", "slower": "faster",
    "higher": "lower", "lower": "higher",
    "more": "less", "less": "more",
    "left": "right", "right": "left",
    "up": "down", "down": "up",
    "longer": "shorter", "shorter": "longer",
    "earlier": "later", "later": "earlier",
    "same": "different", "different": "same",
}


def _direction_match(prediction: str, item: CounterfactualItem) -> bool:
    expected = (item.direction_keyword or "").lower().strip()
    if not expected:
        return False
    text = prediction.lower()
    if expected not in text:
        return False
    # Reject if the opposite keyword also appears (e.g. "longer ... shorter").
    opposite = _OPPOSITES.get(expected)
    return not (opposite and opposite in text)


def _magnitude_match(prediction: str, item: CounterfactualItem) -> bool:
    expected = (item.magnitude_bin or "").lower().strip()
    if not expected:
        return False
    text = prediction.lower()
    for phrase in _MAGNITUDE_PHRASES.get(expected, []):
        if re.search(phrase, text):
            return True
    return False


def counterfactual_delta_accuracy(
    predictions: list[str],
    items: list[CounterfactualItem],
    entailment_threshold: float = 0.5,
) -> dict[str, Any]:
    if len(predictions) != len(items):
        raise ValueError("predictions and items length mismatch")

    n = len(items)
    if n == 0:
        return {
            "overall_accuracy": 0.0,
            "direction_accuracy": 0.0,
            "magnitude_accuracy": 0.0,
            "per_physics_category": {},
            "n_items": 0,
        }

    correct = []
    direction = []
    magnitude = []
    by_category: dict[str, list[int]] = {}

    for pred, item in zip(predictions, items):
        ent = _entailment_prob(pred, item.counterfactual_outcome)
        is_correct = int(ent >= entailment_threshold)
        correct.append(is_correct)
        direction.append(int(_direction_match(pred, item)))
        magnitude.append(int(_magnitude_match(pred, item)))
        cat = item.physics_category if isinstance(item.physics_category, str) else item.physics_category.value
        by_category.setdefault(cat, []).append(is_correct)

    per_cat = {
        cat: {"accuracy": float(np.mean(vals)), "n": len(vals)}
        for cat, vals in by_category.items()
    }
    return {
        "overall_accuracy": float(np.mean(correct)),
        "direction_accuracy": float(np.mean(direction)),
        "magnitude_accuracy": float(np.mean(magnitude)),
        "per_physics_category": per_cat,
        "n_items": n,
    }
