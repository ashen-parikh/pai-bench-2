"""Human-annotation task management.

Light-weight on-disk task queue. Tasks live as JSON files in `<root>/queue/`
and migrate to `<root>/done/` once `n_required` annotations are collected.

This module does not implement the annotator UI — it's the persistence and
aggregation layer that bridges the MLLM annotator's drafts and the
itr_filter.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pai_bench.data.schema import QAItem

logger = logging.getLogger(__name__)


@dataclass
class ReviewTask:
    task_id: str
    candidate: QAItem
    annotations: list[dict] = field(default_factory=list)   # [{annotator_id, answer_idx, notes}]
    status: str = "pending"        # pending | adjudicating | done | rejected

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "candidate": self.candidate.model_dump(mode="json"),
            "annotations": self.annotations,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewTask":
        return cls(
            task_id=data["task_id"],
            candidate=QAItem(**data["candidate"]),
            annotations=list(data.get("annotations", [])),
            status=data.get("status", "pending"),
        )


class HumanReviewQueue:
    def __init__(self, root: Path, n_required: int = 3):
        self.root = Path(root)
        self.queue_dir = self.root / "queue"
        self.done_dir = self.root / "done"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.n_required = n_required

    def enqueue(self, candidate: QAItem) -> ReviewTask:
        task = ReviewTask(task_id=str(uuid.uuid4()), candidate=candidate)
        self._write(task, self.queue_dir)
        return task

    def add_annotation(self, task_id: str, annotator_id: str, answer_idx: int,
                       notes: str = "") -> ReviewTask:
        task = self._read(self.queue_dir / f"{task_id}.json")
        task.annotations.append({
            "annotator_id": annotator_id,
            "answer_idx": int(answer_idx),
            "notes": notes,
        })
        if len(task.annotations) >= self.n_required:
            task.status = "done"
            self._write(task, self.done_dir)
            (self.queue_dir / f"{task_id}.json").unlink(missing_ok=True)
        else:
            self._write(task, self.queue_dir)
        return task

    def collect_for_filtering(self) -> tuple[list[QAItem], dict[str, list[list[int]]]]:
        """Return (candidates, annotations) suitable for itr_filter.filter_by_agreement.

        annotations[item_id] = [[ann1_answer], [ann2_answer], ...]
        """
        candidates: list[QAItem] = []
        annotations: dict[str, list[list[int]]] = {}
        for p in self.done_dir.glob("*.json"):
            task = self._read(p)
            candidates.append(task.candidate)
            annotations[task.candidate.item_id] = [[a["answer_idx"]] for a in task.annotations]
        return candidates, annotations

    def _write(self, task: ReviewTask, dest: Path) -> None:
        (dest / f"{task.task_id}.json").write_text(json.dumps(task.to_dict(), indent=2))

    @staticmethod
    def _read(path: Path) -> ReviewTask:
        return ReviewTask.from_dict(json.loads(Path(path).read_text()))
