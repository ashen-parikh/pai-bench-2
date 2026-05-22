"""EnsembleJudge tests using fake inner judges."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pai_bench.data.schema import Domain, GenerationItem, PhysicsCategory
from pai_bench.judge.base import BaseJudge
from pai_bench.judge.ensemble_judge import EnsembleJudge


class _FakeJudge(BaseJudge):
    """Returns a configured score, optionally raising."""

    def __init__(self, score: float | None, judge_type: str = "fake",
                 violations: list[str] | None = None, raises: bool = False):
        self._score = score
        self.judge_type = judge_type
        self._violations = violations or []
        self._raises = raises

    def score(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        if self._raises:
            raise RuntimeError("boom")
        return {
            "score": self._score,
            "judge_type": self.judge_type,
            "uncertainty": "low",
            "violations": list(self._violations),
        }


def _item() -> GenerationItem:
    return GenerationItem(
        item_id="g1", prompt="p", reference_video_path="/tmp/x",
        domain=Domain.EVERYDAY_PHYSICS,
        physics_category=PhysicsCategory.RIGID_BODY,
        qa_pairs=[],
    )


def _video() -> np.ndarray:
    return np.zeros((4, 16, 16, 3), dtype=np.float32)


def test_median_aggregation_picks_middle_score():
    ens = EnsembleJudge(
        judges=[_FakeJudge(0.2), _FakeJudge(0.6), _FakeJudge(0.9)],
        aggregation="median",
    )
    out = ens.score(_video(), _item())
    assert out["score"] == pytest.approx(0.6)
    assert out["n_judges"] == 3
    assert out["judge_type"] == "ensemble"


def test_inter_judge_disagreement_flagged_above_threshold():
    ens = EnsembleJudge(
        judges=[_FakeJudge(0.1), _FakeJudge(0.9)],
        aggregation="mean",
        disagreement_threshold=0.2,
    )
    out = ens.score(_video(), _item())
    assert out["disagreement_flagged"] is True
    assert out["inter_judge_range"] == pytest.approx(0.8)
    # std for [0.1, 0.9] = 0.4
    assert out["inter_judge_std"] == pytest.approx(0.4)
    assert out["uncertainty"] == "high"


def test_consensus_not_flagged_below_threshold():
    ens = EnsembleJudge(
        judges=[_FakeJudge(0.7), _FakeJudge(0.75), _FakeJudge(0.72)],
        aggregation="mean",
        disagreement_threshold=0.1,
    )
    out = ens.score(_video(), _item())
    assert out["disagreement_flagged"] is False
    assert out["uncertainty"] == "medium"
    assert out["score"] == pytest.approx(np.mean([0.7, 0.75, 0.72]))


def test_failed_judges_are_dropped_not_propagated():
    ens = EnsembleJudge(
        judges=[_FakeJudge(0.5), _FakeJudge(None, raises=True), _FakeJudge(0.7)],
    )
    out = ens.score(_video(), _item())
    assert out["n_judges"] == 2
    assert set(out["per_judge_scores"]) == {0.5, 0.7}


def test_all_judges_failing_returns_unscored():
    ens = EnsembleJudge(judges=[_FakeJudge(None, raises=True)])
    out = ens.score(_video(), _item())
    assert out["score"] is None
    assert out["n_judges"] == 0
    assert out["uncertainty"] == "high"


def test_violations_union_across_judges():
    ens = EnsembleJudge(
        judges=[
            _FakeJudge(0.4, violations=["acceleration_wrong"]),
            _FakeJudge(0.5, violations=["acceleration_wrong", "interpenetration"]),
        ],
        aggregation="mean",
    )
    out = ens.score(_video(), _item())
    assert set(out["violations"]) == {"acceleration_wrong", "interpenetration"}


def test_empty_judges_raises():
    with pytest.raises(ValueError):
        EnsembleJudge(judges=[])


def test_majority_aggregation():
    ens = EnsembleJudge(
        judges=[_FakeJudge(0.4), _FakeJudge(0.7), _FakeJudge(0.8)],
        aggregation="majority",
    )
    out = ens.score(_video(), _item())
    # Two of three above 0.5 -> majority share = 2/3.
    assert out["score"] == pytest.approx(2 / 3)
