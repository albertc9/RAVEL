"""Replay a recorded architecture through supported, versioned lowerings."""

from copy import deepcopy

from ..exceptions import CompatibilityError
from ..manifest import canonical_sha256
from .arithmetic import constant_arithmetic


def replay_design(recorded, fresh, generation, *, chain=None, parameters=None, native=None):
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
    rebuilt = False
    blocks = {block.convolution.id: block for block in chain.blocks} if chain else {}
    for stage in result.get("stages", ()):
        previous = stage.get("arithmetic")
        if not previous:
            continue
        candidates = constant_arithmetic(blocks[previous["operation_id"]], parameters, native,
                                         paired=previous["paired"], dsp_budgets=(previous["dsp_product_budget"],))
        if not candidates:
            raise CompatibilityError("Refreshed parameters no longer satisfy the recorded arithmetic contract; use ordinary conversion")
        stage["arithmetic"] = candidates[0]
        if previous["paired"]:
            result["coefficient_realization"] = candidates[0]
        rebuilt = True
    if rebuilt:
        result["optimization_search"]["evidence_scope"] = "original-selection-before-parameter-refresh"
        result["optimization_search"]["target"]["status"] = "not-reestimated"
    result["resolved_design_sha256"] = canonical_sha256({key: value for key, value in result.items() if key != "resolved_design_sha256"})
    return result
