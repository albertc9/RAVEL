"""Bounded dynamic programming over typed stream endpoints and stage costs."""

from dataclasses import dataclass, field
from math import prod
from typing import Callable

from ..domain.graph import NumericType
from ..domain.temporal import Finding
from .bridges import Bridge, BridgeStrategy, LOSSLESS_BRIDGES
from .arithmetic import ConstantArithmetic
from .schedules import TokenSchedule
from .windows import ArithmeticSchedule, WindowSchedule


@dataclass(frozen=True)
class StreamContract:
    tensor_id: str
    shape: tuple[int, ...]
    numeric: NumericType
    lanes: int
    order: str = "C"
    protocol: str = "blocking-stream"
    reset: str = "per-inference"

    def __post_init__(self):
        if self.lanes <= 0 or not self.shape or any(extent <= 0 for extent in self.shape):
            raise ValueError("Stream lanes and tensor extents must be positive")

    @property
    def values(self) -> int:
        return prod(self.shape)

    @property
    def words(self) -> int:
        return (self.values + self.lanes - 1) // self.lanes

    def to_dict(self) -> dict[str, object]:
        return {"tensor_id": self.tensor_id, "shape": list(self.shape),
                "numeric": self.numeric.to_dict(), "lanes": self.lanes,
                "order": self.order, "words": self.words, "protocol": self.protocol,
                "reset": self.reset, "backpressure": "blocking", "tail_valid_lanes": (self.values - 1) % self.lanes + 1}


@dataclass(frozen=True, order=True)
class Cost:
    cycles: int
    resource: int
    latency: int

    def extend(self, following: "Cost") -> "Cost":
        return Cost(max(self.cycles, following.cycles), self.resource + following.resource,
                    self.latency + following.latency)


@dataclass(frozen=True)
class Candidate:
    id: str
    version: int
    operation_ids: tuple[str, ...]
    input: StreamContract
    output: StreamContract
    cost: Cost
    specialized: bool
    confidence: str = "analytical"
    schedule: TokenSchedule | None = None
    implementation: WindowSchedule | None = None
    arithmetic: ConstantArithmetic | None = None
    arithmetic_schedule: ArithmeticSchedule | None = None

    @property
    def identity(self) -> tuple[str, int, int, str, int]:
        return (self.id, self.version, self.implementation.positions if self.implementation else 0,
                self.arithmetic.proof.graph_sha256 if self.arithmetic else "",
                self.arithmetic_schedule.reuse_factor if self.arithmetic_schedule else 1)

    def to_dict(self) -> dict[str, object]:
        return {**({"arithmetic": self.arithmetic.to_dict()} if self.arithmetic else {}),
                **({"arithmetic_schedule": self.arithmetic_schedule.to_dict()} if self.arithmetic_schedule else {}),
                "strategy": {"id": self.id, "version": self.version},
                "operation_ids": list(self.operation_ids), "input": self.input.to_dict(),
                "output": self.output.to_dict(), "specialized": self.specialized,
                "execution": {"control": "ap_ctrl_hs-dataflow-process", "reset": "discard-in-flight",
                              "source_owner": ("constant-matrix-csd-cse-dsp" if self.arithmetic else
                                               "captured-window-positions" if self.implementation else
                                               "hls4ml-native-latency-v1" if self.id == "hls4ml-temporal-block" else
                                               "compose-top-and-contracts" if self.id == "identity-layout-view" else "legacy-template-adapter"),
                              "storage": ("partitioned-row-history-and-local-window" if self.implementation else
                                          "delegated-native-line-buffers" if self.id == "hls4ml-temporal-block" else
                                          "zero-alias" if self.id == "identity-layout-view" else
                                          "dense-rom-packing-and-accumulator" if self.id == "aria-dense-wide" else "selected-legacy-streaming-plan"),
                              "input_lanes": self.input.lanes, "output_lanes": self.output.lanes},
                "estimate": {"cycles": self.cost.cycles, "resource_proxy": self.cost.resource,
                             "latency": self.cost.latency, "status": "estimated", "confidence": self.confidence},
                **({"schedule": self.schedule.to_dict()} if self.schedule else {}),
                **({"implementation": {**self.implementation.to_dict(),
                                       **({"arithmetic_order": "proven-modular-affine-graph"} if self.arithmetic else {})}}
                   if self.implementation else {})}


@dataclass(frozen=True)
class ChainPlan:
    stages: tuple[Candidate, ...] = ()
    cost: Cost = Cost(0, 0, 0)
    findings: tuple[Finding, ...] = ()
    bridges: tuple[Bridge, ...] = ()

    @property
    def specialized(self) -> bool:
        return any(stage.specialized for stage in self.stages)

    @property
    def conservative_cost(self) -> tuple[float, int, int]:
        unknown = any(stage.confidence not in {"analytical", "calibrated"} for stage in self.stages)
        return (float("inf") if unknown else self.cost.cycles, self.cost.resource, self.cost.latency)

    @property
    def identity(self) -> tuple[tuple[str, int, int, str, int], ...]:
        return tuple(stage.identity for stage in self.stages)


def resolve_chain(domains: tuple[tuple[Candidate, ...], ...], *, max_candidates: int = 32, max_frontier: int = 128, bridge_strategies: tuple[BridgeStrategy, ...] = LOSSLESS_BRIDGES) -> ChainPlan:
    """Keep non-dominated partial plans at each compatible stream endpoint."""
    frontier = [ChainPlan()]
    for domain in domains:
        if len(domain) > max_candidates:
            return ChainPlan(findings=(Finding("planner.candidate_bound", "Declared stage candidate bound exceeded; no candidates were truncated"),))
        extended = []
        for partial in frontier:
            for candidate in sorted(domain, key=lambda item: item.identity):
                bridges = partial.bridges
                cost = partial.cost
                if partial.stages and partial.stages[-1].output != candidate.input:
                    applicable = tuple(bridge for strategy in bridge_strategies
                                       if (bridge := strategy.evaluate(partial.stages[-1].output, candidate.input)) is not None)
                    bridge = min(applicable, key=lambda entry: (entry.cycles, entry.input.lanes + entry.output.lanes, entry.id, entry.version), default=None)
                    if bridge is None:
                        continue
                    bridges = (*bridges, bridge)
                    cost = cost.extend(Cost(bridge.cycles, bridge.input.lanes + bridge.output.lanes, bridge.cycles))
                extended.append(ChainPlan((*partial.stages, candidate), cost.extend(candidate.cost), bridges=bridges))
        frontier = []
        for plan in sorted(extended, key=lambda item: (item.conservative_cost, item.identity)):
            endpoint = plan.stages[-1].output
            dominated = any(previous.stages[-1].output == endpoint and previous.specialized == plan.specialized
                            and all(left <= right for left, right in zip(previous.conservative_cost, plan.conservative_cost)) for previous in frontier)
            if not dominated:
                frontier.append(plan)
        if len(frontier) > max_frontier:
            return ChainPlan(findings=(Finding("planner.frontier_bound", "Declared non-dominated frontier bound exceeded; no plans were truncated"),))
    qualified = [plan for plan in frontier if plan.specialized]
    if not qualified:
        return ChainPlan(findings=(Finding("planner.no_qualified_plan", "No compatible chain containing a RAVEL specialization exists"),))
    return min(qualified, key=lambda item: (item.conservative_cost, item.identity))


@dataclass(frozen=True)
class ChainResolver:
    id: str
    version: int
    evaluator: Callable[..., ChainPlan] = field(compare=False, repr=False)

    def resolve(self, domains, *, bridge_strategies=LOSSLESS_BRIDGES) -> ChainPlan:
        return self.evaluator(domains, bridge_strategies=bridge_strategies)


TEMPORAL_RESOLVER = ChainResolver("bounded-temporal-dp", 1, resolve_chain)
