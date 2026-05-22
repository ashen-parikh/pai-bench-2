"""Pydantic schemas for PAI-Bench 2 data structures.

All persistent objects in the benchmark (items, predictions, scores) round-trip
through this module. Keep field names stable: downstream tooling (annotation
UIs, leaderboards, ITR filtering) reads these via model_dump().
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Domain(str, Enum):
    AUTONOMOUS_VEHICLES = "autonomous_vehicles"
    ROBOTICS = "robotics"
    INDUSTRIAL = "industrial"
    CONSTRUCTION = "construction"
    AGRICULTURE = "agriculture"
    LABORATORY = "laboratory"
    SPORTS_BIOMECHANICS = "sports_biomechanics"
    EVERYDAY_PHYSICS = "everyday_physics"


class PhysicsCategory(str, Enum):
    RIGID_BODY = "rigid_body"
    FLUID = "fluid"
    DEFORMABLE = "deformable"
    CONTACT = "contact"
    THERMAL = "thermal"
    ELECTROMAGNETIC = "electromagnetic"


class QAItem(BaseModel):
    """A multiple-choice QA item used in track U and embedded in track G items."""

    model_config = ConfigDict(use_enum_values=True)

    item_id: str
    video_path: str
    domain: Domain
    physics_category: PhysicsCategory
    question: str
    choices: list[str] = Field(..., min_length=4, max_length=4)
    answer_idx: int = Field(..., ge=0, le=3)
    requires_temporal: bool
    # Items with language_prior_score >= 0.6 are filtered out by the auditor.
    language_prior_score: float = Field(..., ge=0.0, le=1.0)
    # Cohen's kappa across the annotation panel.
    annotator_agreement: float = Field(..., ge=-1.0, le=1.0)
    # IRT 2PL parameters fit on submitted-model response data.
    difficulty: float = 0.0           # b
    discrimination: float = 1.0       # a
    tags: list[str] = Field(default_factory=list)
    # For multi-hop chains, items share a chain_id and have hop_index set.
    chain_id: str | None = None
    hop_index: int | None = None


class GenerationItem(BaseModel):
    """A track-G item: prompt + reference + structured physics expectations."""

    model_config = ConfigDict(use_enum_values=True)

    item_id: str
    prompt: str
    reference_video_path: str
    domain: Domain
    physics_category: PhysicsCategory
    qa_pairs: list[QAItem]
    # Free-form structured expectations consumed by PhysicsVerifier.
    # e.g. {"gravity_direction": [0,-1,0], "expected_collisions": 2}
    expected_physics: dict = Field(default_factory=dict)


class ConditionalItem(BaseModel):
    """A track-C item: prompt + control signals + reference video."""

    model_config = ConfigDict(use_enum_values=True)

    item_id: str
    prompt: str
    # signal_type -> file path (e.g. "depth" -> "data/sample/.../depth.mp4")
    control_signals: dict[str, str]
    reference_video_path: str
    domain: Domain
    variant_prompts: list[str] = Field(default_factory=list)


class CounterfactualItem(BaseModel):
    """A track-CF item: base scenario + intervened variable + expected outcome."""

    model_config = ConfigDict(use_enum_values=True)

    item_id: str
    base_video_path: str
    base_description: str
    counterfactual_variable: str      # e.g. "mass_of_object_A"
    counterfactual_change: str        # e.g. "doubled", "halved"
    base_outcome: str
    counterfactual_outcome: str       # ground-truth answer
    domain: Domain
    physics_category: PhysicsCategory
    # Optional keyword tags for direction / magnitude scoring.
    direction_keyword: str | None = None    # e.g. "faster", "lower"
    magnitude_bin: str | None = None        # one of: much_more, slightly_more,
                                            # same, slightly_less, much_less


class BenchmarkItem(BaseModel):
    """Discriminated union wrapper used by the data loader."""

    track: Literal["G", "C", "U", "CF", "DV"]
    item: GenerationItem | ConditionalItem | QAItem | CounterfactualItem


class ModelPrediction(BaseModel):
    """A single prediction emitted by a model under evaluation."""

    model_id: str
    item_id: str
    track: str
    # For QA: {"answer_idx": int}; for generation: {"video_path": str};
    # for CF: free-form text.
    prediction: str | dict
    confidence: float | None = None
    latency_ms: float | None = None


class TrackScore(BaseModel):
    """Per-track aggregated score for a single model."""

    track: str
    model_id: str
    n_items: int
    scores: dict[str, float]
    per_domain: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_physics_category: dict[str, dict[str, float]] = Field(default_factory=dict)
