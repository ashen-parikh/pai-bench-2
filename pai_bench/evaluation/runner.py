"""Benchmark runner: orchestrates evaluation across tracks.

Writes per-track JSON to runs/{model_id}/{track_id}.json after each track
completes so the runner can be resumed if interrupted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from pai_bench.data.schema import TrackScore
from pai_bench.tracks.base import BaseTrack
from pai_bench.tracks.conditional import ConditionalTrack
from pai_bench.tracks.counterfactual import CounterfactualTrack
from pai_bench.tracks.generation import GenerationTrack
from pai_bench.tracks.understanding import UnderstandingTrack

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run a model across one or more tracks. DV is held-out and not run client-side."""

    PUBLIC_TRACKS = ("G", "C", "U", "CF")

    def __init__(self, config_dir: Path, data_dir: Path):
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.tracks: dict[str, BaseTrack] = {
            "G": GenerationTrack(self.config_dir),
            "C": ConditionalTrack(self.config_dir),
            "U": UnderstandingTrack(self.config_dir),
            "CF": CounterfactualTrack(self.config_dir),
        }
        self.console = Console()

    def _run_dir(self, model_id: str, output_dir: Path) -> Path:
        d = Path(output_dir) / model_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run(
        self,
        model_id: str,
        model_fn: Callable,
        output_dir: Path,
        tracks: list[str] | None = None,
        resume: bool = False,
    ) -> dict[str, TrackScore]:
        tracks = tracks or list(self.PUBLIC_TRACKS)
        run_dir = self._run_dir(model_id, Path(output_dir))
        results: dict[str, TrackScore] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console,
        ) as bar:
            task = bar.add_task("Tracks", total=len(tracks))
            for tid in tracks:
                if tid not in self.tracks:
                    logger.warning("unknown track %s; skipping", tid)
                    bar.advance(task)
                    continue
                cache = run_dir / f"{tid}.json"
                if resume and cache.exists():
                    self.console.log(f"[green]Skipping {tid}: cached at {cache}")
                    results[tid] = TrackScore(**json.loads(cache.read_text()))
                    bar.advance(task)
                    continue
                self.console.log(f"[cyan]Running track {tid}")
                ts = self.tracks[tid].run(model_fn, self.data_dir)
                ts.model_id = model_id
                cache.write_text(ts.model_dump_json(indent=2))
                results[tid] = ts
                bar.advance(task)
        return results

    def resume(self, model_id: str, output_dir: Path) -> dict[str, TrackScore]:
        run_dir = self._run_dir(model_id, Path(output_dir))
        out: dict[str, TrackScore] = {}
        for p in run_dir.glob("*.json"):
            tid = p.stem
            try:
                out[tid] = TrackScore(**json.loads(p.read_text()))
            except Exception as exc:
                logger.warning("failed to load %s: %s", p, exc)
        return out
