"""Immutable evidence for analytical implementation selection."""

from dataclasses import dataclass, field
import hashlib
import json

from .chain import ChainPlan
from .calibration import WINDOW_COST_PROFILE
from .resources import ResourceEstimate, rejection_reasons

SEARCH_CANDIDATE_LIMIT = 32


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
    resources: tuple[ResourceEstimate, ...] = ()
    constraints: dict = field(default_factory=dict)
    generated: int = 0
    evaluated: int = 0
    bound_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "policy": {"id": "aria-stream-search", "version": 1},
            "mode": "analytical",
            "status": "complete" if self.complete else "incomplete",
            "candidate_count": len(self.candidates),
            "exploration": {"generated": self.generated, "evaluated": self.evaluated,
                            "limit": SEARCH_CANDIDATE_LIMIT, "bound_reasons": list(self.bound_reasons)},
            "optimality": "within-enumerated-calibrated-domain" if self.complete and self.selected.stages and any(
                stage.confidence == "calibrated" for stage in self.selected.stages) else "not-claimed",
            "selected_candidate": plan_identity(self.selected) if self.selected.stages else None,
            "constraints": self.constraints,
            "performance_qualification": "not_run",
            "selection_reason": "calibrated-predicted-frame-interval" if any(
                stage.confidence == "calibrated" for stage in self.selected.stages) else "retain-incumbent-without-calibrated-coverage",
            "candidates": [
                {
                    "id": plan_identity(plan),
                    "strategies": [stage.id for stage in plan.stages],
                    "predicted_frame_cycles": plan.cost.cycles,
                    "resources": resource.to_dict(),
                    "rejection_reasons": rejection_reasons(resource, self.constraints) if self.constraints else [],
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
                for plan, resource in zip(self.candidates, self.resources or (ResourceEstimate(),) * len(self.candidates))
            ],
        }
