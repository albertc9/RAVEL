"""Separate resource envelopes and explicit per-core feasibility decisions."""

from dataclasses import dataclass
from math import prod, ceil


# Vitis HLS 2023.2 AvailableResources, KU5P ffvb676 -2-e. BRAM units are 18 Kib.
DEVICE_CAPACITY = {"xcku5p-ffvb676-2-e": {"LUT": 216960, "FF": 433920, "DSP": 1824, "BRAM": 960}}
RESOURCE_NAMES = ("LUT", "FF", "DSP", "BRAM")


@dataclass(frozen=True)
class ResourceEstimate:
    lut: int | None = None
    ff: int | None = None
    dsp: int | None = None
    bram: int | None = None
    policy: str = "conservative-structural-envelope-v1"

    @property
    def values(self):
        return dict(zip(RESOURCE_NAMES, (self.lut, self.ff, self.dsp, self.bram)))

    def to_dict(self):
        return {"values": self.values, "status": "unknown" if self.lut is None else "predicted",
                "units": {"BRAM": "18-Kib-blocks"}, "policy": self.policy}


def estimate_resources(plan, chain) -> ResourceEstimate:
    window = next((s.implementation for s in plan.stages if s.implementation), None)
    if window is None or any(s.confidence not in {"calibrated", "analytical"} for s in plan.stages):
        return ResourceEstimate()
    first = chain.blocks[0].convolution
    first_products = prod(next(p.shape for p in first.parameters if p.role == "weight")) * chain.blocks[0].input.shape[1]
    dense_lanes = plan.stages[-1].cost.resource
    history_bits = window.kernel_rows * window.width * window.channels * plan.stages[1].input.numeric.width
    if any(stage.arithmetic for stage in plan.stages):
        lut, ff, dsp = 6000, 4000 + history_bits, 128
        for stage in plan.stages[:2]:
            if stage.arithmetic:
                arithmetic = stage.arithmetic
                replicas = stage.arithmetic_schedule.engines if stage.arithmetic_schedule else chain.blocks[0].input.shape[1]
                width = arithmetic.accumulator_numeric.width
                adders = sum(arithmetic.count(name) for name in ("add_nodes", "subtract_nodes", "negate_nodes"))
                lut += ceil(0.7 * width * adders * replicas)
                ff += width * (adders + arithmetic.count("output_values")) * replicas
                dsp += arithmetic.count("multiply_nodes") * replicas
            else:
                products = stage.implementation.products if stage.implementation else first_products
                lut += 96 * products
                ff += 32 * products
                dsp += products
        return ResourceEstimate(lut, ff, dsp, 24 + ceil(dense_lanes / 12), "constant-matrix-structural-envelope-v1")
    # Deliberately reserve independent envelopes: no LUT/DSP interchange is used
    # to waive a limit. Native constant folding may consume considerably less.
    return ResourceEstimate(
        8000 + 80 * window.products + 96 * first_products + 32 * dense_lanes,
        4000 + 32 * window.products + history_bits + 16 * first_products,
        window.products + first_products + dense_lanes,
        24 + ceil(window.products / 12),
    )


def constraint_report(part, clock, limits):
    capacity = DEVICE_CAPACITY.get(part, {})
    effective = {name: min(value, limits.get(name, value)) for name, value in capacity.items()}
    effective.update({name: value for name, value in limits.items() if name not in effective})
    return {"part": part, "clock_period_ns": clock, "resource_limits": effective,
            "limits_source": "per-core-and-device-capacity" if limits else "device-capacity" if capacity else "unknown-device-capacity",
            "explicit_limits": dict(limits), "status": "predicted-or-unknown"}


def rejection_reasons(resources, constraints):
    reasons = []
    for name, ceiling in constraints["resource_limits"].items():
        estimate = resources.values[name]
        if estimate is not None and estimate > ceiling:
            reasons.append(f"resource.{name}.exceeds_limit")
        elif estimate is None and name in constraints["explicit_limits"]:
            reasons.append(f"resource.{name}.unknown_under_explicit_limit")
    return reasons
