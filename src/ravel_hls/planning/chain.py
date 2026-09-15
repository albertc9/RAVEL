"""Bounded dynamic programming over typed stream endpoints and stage costs."""

from dataclasses import dataclass
from math import prod

from ..domain.graph import NumericType
from ..domain.temporal import Finding
from .bridges import Bridge, bridge_for


@dataclass(frozen=True)
class StreamContract:
    tensor_id: str
    shape: tuple[int, ...]
    numeric: NumericType
    lanes: int
    order: str = "C"
    protocol: str = "blocking-stream"
    reset: str = "per-inference"

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

    @property
    def identity(self) -> tuple[str, int]:
        return self.id, self.version

    def to_dict(self) -> dict[str, object]:
        return {"strategy": {"id": self.id, "version": self.version},
                "operation_ids": list(self.operation_ids), "input": self.input.to_dict(),
                "output": self.output.to_dict(), "specialized": self.specialized,
                "estimate": {"cycles": self.cost.cycles, "resource_proxy": self.cost.resource,
                             "latency": self.cost.latency, "status": "estimated", "confidence": self.confidence}}


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
    def identity(self) -> tuple[tuple[str, int], ...]:
        return tuple(stage.identity for stage in self.stages)


def resolve_chain(domains: tuple[tuple[Candidate, ...], ...], *, max_candidates: int = 32, max_frontier: int = 128) -> ChainPlan:
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
                    bridge = bridge_for(partial.stages[-1].output, candidate.input)
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
