"""JSON / Markdown report writers."""

from __future__ import annotations

import json
from pathlib import Path

from pai_bench.data.schema import TrackScore


def write_json(scores: dict[str, TrackScore], path: Path) -> None:
    payload = {tid: ts.model_dump(mode="json") for tid, ts in scores.items()}
    Path(path).write_text(json.dumps(payload, indent=2))


def write_markdown(scores: dict[str, TrackScore], path: Path) -> None:
    lines = ["# PAI-Bench 2 evaluation report", ""]
    for tid, ts in scores.items():
        lines.append(f"## Track {tid} ({ts.model_id})")
        lines.append("")
        lines.append(f"- n_items: {ts.n_items}")
        for k, v in ts.scores.items():
            lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, (int, float)) else f"- **{k}**: {v}")
        if ts.per_domain:
            lines.append("")
            lines.append("### Per-domain")
            for dom, metrics in ts.per_domain.items():
                lines.append(f"- {dom}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        if ts.per_physics_category:
            lines.append("")
            lines.append("### Per-physics-category")
            for cat, metrics in ts.per_physics_category.items():
                lines.append(f"- {cat}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        lines.append("")
    Path(path).write_text("\n".join(lines))
