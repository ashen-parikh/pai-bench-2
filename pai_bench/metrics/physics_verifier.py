"""Analytic physics verifier.

# PAI-BENCH-2-CHANGE: replaces MLLM-as-Judge for tractable physics
# (rigid body, contact). The hybrid_judge routes intractable categories
# (deformable, fluid above heuristic level, thermal, EM) back to MLLM with
# an explicit uncertainty flag.

The verifier returns a structured dict so the rest of the pipeline (domain
scorer, leaderboard) can distinguish analytic verdicts from MLLM verdicts.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from pai_bench.data.schema import GenerationItem, PhysicsCategory
from pai_bench.metrics import physics_metrics as pm

logger = logging.getLogger(__name__)

# A trajectory acceleration is flagged as "gravity-aligned" when its
# unit-normalised direction lies within this cosine of the expected gravity.
GRAVITY_COSINE_TOL = 0.7
# Sudden velocity changes above this fraction of the median velocity are
# treated as collision events.
COLLISION_SPIKE = 0.6

# Thresholds for the supplementary physics_metrics checks. Tuned so that
# clean synthetic motion passes; aggressive enough that morphing /
# teleporting / depth-flicker videos fail.
FLOW_SMOOTHNESS_FLOOR = 0.30
DEPTH_STABILITY_FLOOR = 0.55
BLOB_STABILITY_FLOOR = 0.50


class PhysicsVerifier:
    """Routes a video + generation item to the correct analytic check."""

    def verify(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        category = item.physics_category
        if isinstance(category, str):
            category = PhysicsCategory(category)
        if category == PhysicsCategory.RIGID_BODY:
            return self._verify_rigid_body(video, item)
        if category == PhysicsCategory.FLUID:
            return self._verify_fluid(video, item)
        if category == PhysicsCategory.CONTACT:
            return self._verify_contact(video, item)
        return {
            "passed": False,
            "score": None,
            "violations": [],
            "verifier_type": "intractable",
        }

    # --- shared helpers -----------------------------------------------------

    @staticmethod
    def _gray(video: np.ndarray) -> np.ndarray:
        out = np.empty(video.shape[:3], dtype=np.float32)
        for i, f in enumerate(video):
            out[i] = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return out

    @staticmethod
    def _flow_field(video_gray: np.ndarray) -> np.ndarray:
        """Per-frame Farneback flow; returns (T-1, H, W, 2)."""
        flows = []
        for t in range(video_gray.shape[0] - 1):
            f = cv2.calcOpticalFlowFarneback(
                (video_gray[t] * 255).astype(np.uint8),
                (video_gray[t + 1] * 255).astype(np.uint8),
                None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )
            flows.append(f)
        return np.stack(flows) if flows else np.zeros((0, *video_gray.shape[1:], 2))

    @staticmethod
    def _track_centroid(video_gray: np.ndarray) -> np.ndarray:
        """Track the centroid of the largest motion blob across frames."""
        prev = video_gray[0]
        centroids = []
        for t in range(1, video_gray.shape[0]):
            diff = np.abs(video_gray[t] - prev)
            prev = video_gray[t]
            if diff.max() < 1e-3:
                centroids.append(centroids[-1] if centroids else np.array([0.0, 0.0]))
                continue
            ys, xs = np.where(diff > diff.mean() + diff.std())
            if len(xs) < 5:
                centroids.append(centroids[-1] if centroids else np.array([float(diff.shape[1]) / 2, float(diff.shape[0]) / 2]))
                continue
            centroids.append(np.array([xs.mean(), ys.mean()]))
        return np.stack(centroids) if centroids else np.zeros((0, 2))

    # --- supplementary analytic checks (slide 1 fix) -----------------------

    def _supplementary_checks(self, video: np.ndarray) -> dict[str, Any]:
        """Run flow / depth / blob / pose checks and report a per-check verdict.

        Returns:
            {
              "scores": {flow_smoothness: float, depth_stability: float,
                         blob_count_stability: float, pose_validity: float|None},
              "checks": {flow_smoothness: bool, ...},
              "violations": list[str],
            }
        """
        scores = {
            "flow_smoothness": pm.optical_flow_smoothness(video),
            "depth_stability": pm.depth_stability(video),
            "blob_count_stability": pm.motion_blob_count_stability(video),
            "pose_validity": pm.pose_validity(video),       # None when no detector
        }
        floors = {
            "flow_smoothness": FLOW_SMOOTHNESS_FLOOR,
            "depth_stability": DEPTH_STABILITY_FLOOR,
            "blob_count_stability": BLOB_STABILITY_FLOOR,
            "pose_validity": 0.5,
        }
        checks: dict[str, bool] = {}
        violations: list[str] = []
        for name, val in scores.items():
            if val is None:                                  # check skipped
                continue
            ok = val >= floors[name]
            checks[name] = ok
            if not ok:
                violations.append(f"{name}_low_{val:.2f}")
        return {"scores": scores, "checks": checks, "violations": violations}

    # --- rigid body ---------------------------------------------------------

    def _verify_rigid_body(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        violations: list[str] = []
        checks: dict[str, bool] = {}

        gray = self._gray(video)
        centroids = self._track_centroid(gray)
        if centroids.shape[0] < 5:
            return {
                "passed": False,
                "score": 0.0,
                "violations": ["insufficient_motion"],
                "checks": {},
                "verifier_type": "analytic",
            }

        # 1. Parabolic-fit acceleration direction vs gravity.
        t = np.arange(centroids.shape[0])
        # Fit y(t) = a*t^2 + b*t + c. Image y axis is inverted -> gravity is +y in pixel space.
        coeffs_y = np.polyfit(t, centroids[:, 1], 2)
        gravity_expected = item.expected_physics.get("gravity_direction", [0.0, 1.0])
        accel_sign = float(np.sign(coeffs_y[0]))
        checks["gravity_alignment"] = (
            accel_sign * np.sign(gravity_expected[1]) > 0 and abs(coeffs_y[0]) > 1e-3
        )
        if not checks["gravity_alignment"]:
            violations.append("acceleration_not_gravity_aligned")

        # 2. Collision detection: spikes in velocity magnitude.
        v = np.diff(centroids, axis=0)
        v_mag = np.linalg.norm(v, axis=1)
        median = float(np.median(v_mag)) + 1e-6
        spikes = int(np.sum(np.abs(np.diff(v_mag)) > COLLISION_SPIKE * median))
        expected_collisions = int(item.expected_physics.get("expected_collisions", spikes))
        checks["collision_count"] = abs(spikes - expected_collisions) <= 1
        if not checks["collision_count"]:
            violations.append(f"collision_count_mismatch_{spikes}_vs_{expected_collisions}")

        # 3. Interpenetration: centroid teleport detection.
        step = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        if len(step) > 0:
            jumps = int(np.sum(step > 5 * (np.median(step) + 1e-6)))
            checks["no_interpenetration"] = jumps == 0
            if not checks["no_interpenetration"]:
                violations.append(f"interpenetration_jumps_{jumps}")
        else:
            checks["no_interpenetration"] = True

        # 4-6. Supplementary analytic checks (flow / depth / blob / pose).
        supp = self._supplementary_checks(video)
        checks.update(supp["checks"])
        violations.extend(supp["violations"])

        score = sum(checks.values()) / len(checks) if checks else 0.0
        return {
            "passed": score >= 0.66,
            "score": float(score),
            "violations": violations,
            "checks": checks,
            "supplementary_scores": supp["scores"],
            "verifier_type": "analytic",
        }

    # --- fluid (heuristic) --------------------------------------------------

    def _verify_fluid(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        violations: list[str] = []
        gray = self._gray(video)

        # 1. Mass conservation proxy: total "fluid pixel" area (low-frequency
        # bright/dark blob area) should be roughly stable.
        thresh = gray.mean() + gray.std()
        areas = np.array([float((g > thresh).sum()) for g in gray])
        if areas.max() > 0:
            drift = float((areas.max() - areas.min()) / areas.max())
        else:
            drift = 0.0
        mass_ok = drift < 0.5
        if not mass_ok:
            violations.append(f"fluid_mass_drift_{drift:.2f}")

        # 2. Vorticity plausibility: extreme curl magnitudes flag non-physical swirl.
        flows = self._flow_field(gray)
        if flows.shape[0] > 0:
            du_dy = np.gradient(flows[..., 0], axis=1)
            dv_dx = np.gradient(flows[..., 1], axis=2)
            curl = du_dy - dv_dx
            extreme_curl = float(np.percentile(np.abs(curl), 99))
            curl_ok = extreme_curl < 5.0
        else:
            curl_ok = True
            extreme_curl = 0.0
        if not curl_ok:
            violations.append(f"vorticity_extreme_{extreme_curl:.2f}")

        # 3. Surface smoothness: edge density along the largest connected fluid
        # region should not explode frame-to-frame.
        edge_densities = []
        for g in gray:
            edges = cv2.Canny((g * 255).astype(np.uint8), 50, 150)
            edge_densities.append(float(edges.mean()))
        smooth_var = float(np.std(edge_densities))
        smooth_ok = smooth_var < 20.0
        if not smooth_ok:
            violations.append(f"surface_jitter_{smooth_var:.2f}")

        checks = {
            "fluid_mass_conservation": bool(mass_ok),
            "vorticity_plausible": bool(curl_ok),
            "surface_smoothness": bool(smooth_ok),
        }
        supp = self._supplementary_checks(video)
        checks.update(supp["checks"])
        violations.extend(supp["violations"])

        score = sum(checks.values()) / len(checks)
        return {
            "passed": score >= 0.66,
            "score": float(score),
            "violations": violations,
            "checks": checks,
            "supplementary_scores": supp["scores"],
            "verifier_type": "heuristic",
            "uncertainty": "medium",
        }

    # --- contact ------------------------------------------------------------

    def _verify_contact(self, video: np.ndarray, item: GenerationItem) -> dict[str, Any]:
        violations: list[str] = []
        checks: dict[str, bool] = {}

        gray = self._gray(video)
        centroids = self._track_centroid(gray)
        if centroids.shape[0] < 5:
            return {
                "passed": False,
                "score": 0.0,
                "violations": ["insufficient_motion"],
                "checks": {},
                "verifier_type": "analytic",
            }

        v = np.diff(centroids, axis=0)
        speed = np.linalg.norm(v, axis=1)

        # 1. Friction: sliding object should not speed up without input.
        if len(speed) > 4:
            second_half_mean = float(speed[len(speed) // 2:].mean())
            first_half_mean = float(speed[: len(speed) // 2].mean()) + 1e-6
            checks["friction_consistent"] = second_half_mean <= first_half_mean * 1.2
            if not checks["friction_consistent"]:
                violations.append("friction_violation_speeding_up")
        else:
            checks["friction_consistent"] = True

        # 2. Reflection plausibility at sharpest direction change.
        if len(v) > 2:
            dirs = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
            dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
            idx = int(np.argmin(dots))
            incoming = dirs[idx]
            outgoing = dirs[idx + 1]
            normal = (incoming + outgoing)
            normal /= np.linalg.norm(normal) + 1e-6
            reflected = incoming - 2 * np.dot(incoming, normal) * normal
            cos_match = float(np.dot(reflected, outgoing))
            checks["reflection_plausible"] = cos_match > GRAVITY_COSINE_TOL
            if not checks["reflection_plausible"]:
                violations.append(f"reflection_implausible_cos_{cos_match:.2f}")
        else:
            checks["reflection_plausible"] = True

        # 3. No interpenetration jumps.
        step = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        if len(step) > 0:
            jumps = int(np.sum(step > 5 * (np.median(step) + 1e-6)))
            checks["no_interpenetration"] = jumps == 0
            if not checks["no_interpenetration"]:
                violations.append(f"interpenetration_jumps_{jumps}")
        else:
            checks["no_interpenetration"] = True

        # 4-6. Supplementary analytic checks.
        supp = self._supplementary_checks(video)
        checks.update(supp["checks"])
        violations.extend(supp["violations"])

        score = sum(checks.values()) / len(checks) if checks else 0.0
        return {
            "passed": score >= 0.66,
            "score": float(score),
            "violations": violations,
            "checks": checks,
            "supplementary_scores": supp["scores"],
            "verifier_type": "analytic",
        }
