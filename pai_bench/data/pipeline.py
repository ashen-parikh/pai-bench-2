"""End-to-end annotation pipeline orchestration.

Composes mllm_annotator → human_review → language_prior → itr_filter → irt
into a single callable used by the dataset-build scripts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pai_bench.annotation.human_review import HumanReviewQueue
from pai_bench.annotation.irt_calibration import IRTCalibrator
from pai_bench.annotation.itr_filter import filter_by_agreement
from pai_bench.annotation.language_prior import LanguagePriorAuditor
from pai_bench.annotation.mllm_annotator import MLLMAnnotator
from pai_bench.data.schema import QAItem

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    n_drafted: int
    n_after_review: int
    n_after_language_prior: int
    n_after_agreement: int
    irt_flags: dict


class AnnotationPipeline:
    def __init__(
        self,
        review_root: Path,
        annotator: MLLMAnnotator | None = None,
        language_prior: LanguagePriorAuditor | None = None,
        irt: IRTCalibrator | None = None,
    ):
        self.queue = HumanReviewQueue(review_root)
        self.annotator = annotator or MLLMAnnotator()
        self.language_prior = language_prior or LanguagePriorAuditor()
        self.irt = irt or IRTCalibrator()

    def run(
        self,
        seeds: list[dict],
        irt_responses: np.ndarray | None = None,
        agreement_threshold: float = 0.80,
    ) -> tuple[list[QAItem], PipelineReport]:
        drafted: list[QAItem] = []
        for s in seeds:
            drafts = self.annotator.draft(
                video_path=Path(s["video_path"]),
                domain=s["domain"],
                physics_category=s["physics_category"],
                ontology_branch=s.get("ontology_branch", ""),
                n_candidates=s.get("n_candidates", 3),
            )
            drafted.extend(drafts)
            for d in drafts:
                self.queue.enqueue(d)

        # Skipping the actual human-in-the-loop step here; caller is expected
        # to drop annotations into the queue before invoking the rest.
        reviewed, annotations = self.queue.collect_for_filtering()

        passed_lp = self.language_prior.audit_batch(reviewed)
        annotations_filtered = {i.item_id: annotations[i.item_id] for i in passed_lp}
        passed_agreement, _flagged = filter_by_agreement(
            passed_lp, annotations_filtered, threshold=agreement_threshold,
        )

        irt_flags: dict = {}
        if irt_responses is not None and len(passed_agreement) == irt_responses.shape[1]:
            a, b = self.irt.fit(irt_responses)
            for idx, item in enumerate(passed_agreement):
                item.discrimination = float(a[idx])
                item.difficulty = float(b[idx])
            irt_flags = self.irt.flag_items(a, b)

        report = PipelineReport(
            n_drafted=len(drafted),
            n_after_review=len(reviewed),
            n_after_language_prior=len(passed_lp),
            n_after_agreement=len(passed_agreement),
            irt_flags=irt_flags,
        )
        return passed_agreement, report
