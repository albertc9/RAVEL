"""Replay a recorded architecture through supported, versioned lowerings."""

from copy import deepcopy

from ..exceptions import CompatibilityError
from ..manifest import canonical_sha256


def replay_design(recorded, fresh, generation):
    supported = {(entry.id, entry.version) for entry in generation.stage_strategies}
    for stage in recorded.get("stages", ()):
        identity = stage["strategy"]
        if (identity["id"], identity["version"]) not in supported:
            raise CompatibilityError("Recorded stage lowering is unavailable; use ordinary conversion")
    expected = [(p["id"], p["descriptor"]) for p in recorded["parameter_bindings"]]
    observed = [(p["id"], p["descriptor"]) for p in fresh["parameter_bindings"]]
    if observed != expected:
        raise CompatibilityError("Refresh changes recorded parameter descriptors; use ordinary conversion")
    result = deepcopy(recorded)
    result["parameter_bindings"] = deepcopy(fresh["parameter_bindings"])
    if "coefficient_realization" in fresh:
        result["coefficient_realization"] = deepcopy(fresh["coefficient_realization"])
    result["resolved_design_sha256"] = canonical_sha256({key: value for key, value in result.items() if key != "resolved_design_sha256"})
    return result
