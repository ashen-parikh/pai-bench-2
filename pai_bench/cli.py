"""PAI-Bench CLI entry point.

  pai-bench run         — execute a model across one or more tracks
  pai-bench score       — recompute scores from cached predictions
  pai-bench audit       — language-prior / agreement / IRT / saturation checks
  pai-bench leaderboard — aggregate multiple run directories
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler

from pai_bench.annotation.irt_calibration import IRTCalibrator
from pai_bench.annotation.itr_filter import filter_by_agreement
from pai_bench.annotation.language_prior import LanguagePriorAuditor
from pai_bench.data.loader import load_items
from pai_bench.evaluation.leaderboard import Leaderboard
from pai_bench.evaluation.report import write_json, write_markdown
from pai_bench.evaluation.runner import BenchmarkRunner
from pai_bench.evaluation.scorer import TRACK_CLASSES, rescore_track
from pai_bench.qc.saturation import saturation_report

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _import_model_fn(spec: str):
    """Import a callable from a "module:attr" specifier."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise click.UsageError(f"model spec must be 'module:callable'; got {spec!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise click.UsageError(f"{spec} is not callable")
    return fn


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def main(verbose: bool) -> None:
    _setup_logging(verbose)


@main.command()
@click.option("--model-id", required=True)
@click.option("--model-config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--tracks", default="G,C,U,CF",
              help="Comma-separated track IDs.")
@click.option("--data-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--resume/--no-resume", default=False)
def run(model_id: str, model_config: Path, tracks: str, data_dir: Path,
        output_dir: Path, resume: bool) -> None:
    """Run a model across the specified tracks."""
    cfg = yaml.safe_load(Path(model_config).read_text())
    fn_spec = cfg.get("model_fn")
    if not fn_spec:
        raise click.UsageError("model-config must contain a 'model_fn' field")
    model_fn = _import_model_fn(fn_spec)
    tracks_list = [t.strip() for t in tracks.split(",") if t.strip()]

    config_dir = Path(__file__).resolve().parent.parent / "config"
    runner = BenchmarkRunner(config_dir, data_dir)
    scores = runner.run(model_id, model_fn, Path(output_dir),
                        tracks=tracks_list, resume=resume)
    summary = {tid: ts.scores for tid, ts in scores.items()}
    console.print_json(json.dumps(summary, indent=2))


@main.command()
@click.option("--run-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["json", "md", "both"]), default="both")
def score(run_dir: Path, fmt: str) -> None:
    """Render JSON / Markdown reports from a completed run directory."""
    from pai_bench.data.schema import TrackScore
    scores: dict[str, TrackScore] = {}
    for p in Path(run_dir).glob("*.json"):
        if p.name in {"report.json", "report.md"}:
            continue
        try:
            ts = TrackScore(**json.loads(p.read_text()))
        except Exception as exc:
            console.log(f"[yellow]Skipping {p}: {exc}")
            continue
        scores[ts.track] = ts
    if fmt in ("json", "both"):
        write_json(scores, Path(run_dir) / "report.json")
    if fmt in ("md", "both"):
        write_markdown(scores, Path(run_dir) / "report.md")
    console.log(f"[green]Wrote report to {run_dir}")


@main.command()
@click.option("--data-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--check",
              type=click.Choice(["language_prior", "agreement", "irt", "saturation", "all"]),
              default="all")
def audit(data_dir: Path, check: str) -> None:
    """Run benchmark QC checks."""
    items = load_items(Path(data_dir), "U")
    qa_items = [bi.item for bi in items]

    if check in ("language_prior", "all"):
        flagged = [i for i in qa_items if i.language_prior_score >= 0.6]
        console.log(f"language_prior: {len(flagged)}/{len(qa_items)} items at or above threshold")
    if check in ("agreement", "all"):
        flagged = [i for i in qa_items if i.annotator_agreement < 0.80]
        console.log(f"agreement: {len(flagged)}/{len(qa_items)} items below kappa threshold")
    if check in ("irt", "all"):
        cal = IRTCalibrator()
        scores_path = Path(data_dir) / "irt_responses.json"
        if scores_path.exists():
            import numpy as np
            resp = np.array(json.loads(scores_path.read_text()))
            a, b = cal.fit(resp)
            flags = cal.flag_items(a, b)
            console.log(f"irt: floor={len(flags['floor'])} ceiling={len(flags['ceiling'])} low_disc={len(flags['low_disc'])}")
        else:
            console.log(f"irt: no irt_responses.json in {data_dir}; skipping")
    if check in ("saturation", "all"):
        scores_path = Path(data_dir) / "item_scores.json"
        if scores_path.exists():
            item_scores = json.loads(scores_path.read_text())
            report = saturation_report(items, item_scores)
            console.print_json(json.dumps(report, indent=2))
        else:
            console.log(f"saturation: no item_scores.json in {data_dir}; skipping")


@main.command()
@click.option("--results-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
def leaderboard(results_dir: Path, output: Path) -> None:
    """Aggregate all model run directories under results-dir into a leaderboard."""
    from pai_bench.data.schema import TrackScore
    lb = Leaderboard()
    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir():
            continue
        scores: dict[str, TrackScore] = {}
        for p in model_dir.glob("*.json"):
            try:
                ts = TrackScore(**json.loads(p.read_text()))
                scores[ts.track] = ts
            except Exception:
                continue
        if scores:
            lb.add_model(model_dir.name, scores)
    lb.to_markdown(Path(output))
    console.log(f"[green]Wrote leaderboard to {output}")


if __name__ == "__main__":
    main()
