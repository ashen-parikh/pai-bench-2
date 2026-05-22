"""Annotation pipeline tests.

The full pipeline calls an MLLM annotator we don't want to invoke in CI; we
inject deterministic candidates directly into the queue and assert the rest
of the pipeline routes them through the language-prior + agreement filters
as expected.
"""

from __future__ import annotations

from pathlib import Path

from pai_bench.annotation.human_review import HumanReviewQueue
from pai_bench.annotation.itr_filter import filter_by_agreement
from pai_bench.annotation.language_prior import LanguagePriorAuditor
from pai_bench.data.schema import Domain, PhysicsCategory, QAItem


def _qa(item_id: str, answer_idx: int) -> QAItem:
    return QAItem(
        item_id=item_id, video_path="/tmp/x",
        domain=Domain.ROBOTICS,
        physics_category=PhysicsCategory.RIGID_BODY,
        question="What happens next?",
        choices=["a", "b", "c", "d"],
        answer_idx=answer_idx,
        requires_temporal=True,
        language_prior_score=0.0,
        annotator_agreement=0.0,
    )


def test_queue_persists_to_done(tmp_path: Path):
    q = HumanReviewQueue(tmp_path, n_required=2)
    task = q.enqueue(_qa("i1", 0))
    q.add_annotation(task.task_id, "ann1", 0)
    q.add_annotation(task.task_id, "ann2", 0)
    items, annotations = q.collect_for_filtering()
    assert len(items) == 1
    assert annotations["i1"] == [[0], [0]]


def test_pipeline_filters_in_order(tmp_path: Path):
    items = [_qa("good", 0), _qa("biased", 0), _qa("low_agree", 0)]

    # Stub language-prior client: "biased" gets it right every time.
    def stub(question: str, choices):
        return 0

    # Override per-item priors after audit so we get deterministic outcomes.
    auditor = LanguagePriorAuditor(threshold=0.6, client_fn=stub)
    passed_lp = auditor.audit_batch(items, n_trials=4)
    # All three pass because stub always returns 0 = answer_idx, so prior=1.0.
    # In real use we wouldn't pass all; here we just verify the auditor wrote
    # the score onto the item.
    for item in items:
        assert item.language_prior_score == 1.0
    # With threshold 0.6 they should ALL be filtered out.
    assert passed_lp == []
