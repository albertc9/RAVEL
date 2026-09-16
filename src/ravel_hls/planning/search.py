"""Immutable evidence for analytical implementation selection."""

from dataclasses import dataclass
import hashlib
import json

from .chain import ChainPlan
from .calibration import WINDOW_COST_PROFILE


def plan_identity(plan: ChainPlan) -> str:
    payload = {
        "stages": [stage.to_dict() for stage in plan.stages],
        "bridges": [bridge.to_dict() for bridge in plan.bridges],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SearchReport:
    """Candidate decisions, separate from source-bound vendor qualification."""

    candidates: tuple[ChainPlan, ...]
    selected: ChainPlan
    complete: bool = True

    def to_dict(self) -> dict:
        return {
            "policy": {"id": "aria-stream-search", "version": 1},
            "mode": "analytical",
            "status": "complete" if self.complete else "incomplete",
            "candidate_count": len(self.candidates),
            "selected_candidate": plan_identity(self.selected),
            "performance_qualification": "not_run",
            "selection_reason": "calibrated-predicted-frame-interval" if any(
                stage.confidence == "calibrated" for stage in self.selected.stages) else "retain-incumbent-without-calibrated-coverage",
            "candidates": [
                {
                    "id": plan_identity(plan),
                    "strategies": [stage.id for stage in plan.stages],
                    "predicted_frame_cycles": plan.cost.cycles,
                    "schedule": next((stage.implementation.to_dict() for stage in plan.stages
                                      if stage.implementation is not None), None),
                    "confidence": (
                        "calibrated" if any(stage.confidence == "calibrated" for stage in plan.stages)
                        else "analytical" if all(stage.confidence == "analytical" for stage in plan.stages)
                        else "uncalibrated"
                    ),
                    "calibration_profile": WINDOW_COST_PROFILE.to_dict() if any(
                        stage.confidence == "calibrated" for stage in plan.stages) else None,
                }
                for plan in self.candidates
            ],
        }
