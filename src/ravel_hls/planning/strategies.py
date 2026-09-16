"""Closed, pure strategy capabilities over immutable semantic stage facts."""

from dataclasses import dataclass, field, replace
from math import prod
from typing import Callable

from ..domain.temporal import Finding, LayoutView, TemporalBlock
from ..domain.graph import OperationFacts
from .chain import Candidate, Cost, StreamContract
from .schedules import TokenSchedule, temporal_schedule
from .windows import window_schedules
from .calibration import WINDOW_COST_PROFILE, CONSTANT_MATRIX_COST_PROFILE


@dataclass(frozen=True)
class StageContext:
    external_endpoint: str
    input_strategy: str
    temporal_packing: int
    dense_parallelism: int
    input_cycles: int
    dense_cycles: int
    part: str | None = None
    clock_period: float | None = None
    arithmetic: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StageRequest:
    semantic: TemporalBlock | LayoutView | OperationFacts
    context: StageContext
    previous: StreamContract | None = None
    layout: LayoutView | None = None


@dataclass(frozen=True)
class Capability:
    candidates: tuple[Candidate, ...] = ()
    findings: tuple[Finding, ...] = ()

    @property
    def applicable(self) -> bool:
        return bool(self.candidates)


@dataclass(frozen=True)
class StageStrategy:
    id: str
    version: int
    evaluator: Callable[[StageRequest, str, int], Capability] = field(compare=False, repr=False)

    def evaluate(self, request: StageRequest) -> Capability:
        return self.evaluator(request, self.id, self.version)

    def to_dict(self) -> dict:
        return {"id": self.id, "version": self.version}


def _temporal(request, strategy, version):
    block, context = request.semantic, request.context
    if not isinstance(block, TemporalBlock):
        return Capability()
    external = block.input.id == context.external_endpoint
    specialized = strategy != "hls4ml-temporal-block"
    if external != specialized or (specialized and strategy != context.input_strategy):
        return Capability()
    convolution, pooled = block.convolution, block.pooling.outputs[0]
    input_lanes = int(convolution.attribute("n_chan"))
    output_lanes = int(convolution.attribute("n_filt"))
    if specialized:
        input_lanes *= block.input.shape[1] * context.temporal_packing
        output_lanes *= pooled.shape[1]
    source = StreamContract(block.input.id, block.input.shape, block.input.numeric_type, input_lanes)
    target = StreamContract(pooled.id, pooled.shape, pooled.numeric_type, output_lanes)
    weight = next((parameter for parameter in convolution.parameters if parameter.role == "weight"), None)
    multipliers = prod(weight.shape) if weight is not None else 0
    for operation in (block.convolution, block.pooling):
        if any(operation.attribute(name, 0) != 0 for name in ("pad_top", "pad_bottom", "pad_left", "pad_right")):
            return Capability(findings=(Finding("strategy.geometry.padding", "The qualified temporal schedule requires valid unpadded windows", operation.id),))
    try:
        schedule = temporal_schedule(block, input_lanes, output_lanes)
    except ValueError as error:
        return Capability(findings=(Finding("strategy.schedule.event_bound", str(error), convolution.id),))
    lower_bound = max(source.words, target.words)
    base = Candidate(strategy, version, (convolution.id, block.activation.id, block.pooling.id), source, target,
                                Cost(context.input_cycles if specialized else lower_bound, multipliers, lower_bound), specialized,
                                "analytical" if specialized else "uncalibrated-native", schedule)
    variants = tuple(replace(base, arithmetic=arithmetic,
                            cost=Cost(context.input_cycles + 5, multipliers, context.input_cycles + 5),
                            confidence="calibrated" if CONSTANT_MATRIX_COST_PROFILE.covers(
                                block, arithmetic, context.part, context.clock_period) else "uncalibrated")
                     for arithmetic in context.arithmetic.get(convolution.id, ())
                     if strategy == "phara")
    return Capability((base, *variants))


def _layout(request, strategy, version):
    if not isinstance(request.semantic, LayoutView) or request.previous is None:
        return Capability()
    view, source = request.semantic, request.previous
    if source.tensor_id != view.input.id or not view.input.numeric_type.preserves_codes_in(view.output.numeric_type):
        return Capability(findings=(Finding("strategy.layout.codes", "Layout view does not preserve the endpoint's integer codes", view.operation.id),))
    output = StreamContract(view.output.id, view.output.shape, view.output.numeric_type, source.lanes)
    return Capability((Candidate(strategy, version, (view.operation.id,), source, output, Cost(0, 0, 0), False,
                                schedule=TokenSchedule(source.words, output.words, tuple(range(1, output.words + 1)))),))


def _scheduled(request, strategy, version):
    block = request.semantic
    if not isinstance(block, TemporalBlock) or block.input.id == request.context.external_endpoint:
        return Capability()
    candidates = []
    findings = []
    for implementation in window_schedules(block):
        calibrated = WINDOW_COST_PROFILE.covers(block, implementation, request.context.part, request.context.clock_period)
        pooled = block.pooling.outputs[0]
        source = StreamContract(block.input.id, block.input.shape, block.input.numeric_type,
                                implementation.positions * implementation.channels)
        target = StreamContract(pooled.id, pooled.shape, pooled.numeric_type,
                                implementation.positions * implementation.filters)
        try:
            schedule = temporal_schedule(block, source.lanes, target.lanes)
        except ValueError as error:
            findings.append(Finding("strategy.schedule.event_bound", str(error), block.convolution.id))
            continue
        candidates.append(Candidate(
            strategy, version,
            (block.convolution.id, block.activation.id, block.pooling.id), source, target,
            Cost(WINDOW_COST_PROFILE.window_cycles(implementation) if calibrated else implementation.input_words,
                 implementation.products, implementation.input_words),
            True, "calibrated" if calibrated else "uncalibrated", schedule, implementation,
        ))
    return Capability(tuple(candidates), tuple(findings))


def _dense(request, strategy, version):
    head, layout, context = request.semantic, request.layout, request.context
    if not isinstance(head, OperationFacts) or head.kind != "dense" or layout is None:
        return Capability()
    lanes = layout.input.shape[-1] * layout.input.shape[-2]
    source = StreamContract(layout.output.id, layout.output.shape, layout.output.numeric_type, lanes)
    output = head.outputs[0]
    target = StreamContract(output.id, output.shape, output.numeric_type, 1)
    return Capability((Candidate(strategy, version, (head.id,), source, target,
                                Cost(context.dense_cycles, layout.input.shape[-1] * context.dense_parallelism, context.dense_cycles), True,
                                schedule=TokenSchedule(source.words, target.words, (source.words,) * target.words)),))


def _affine(request, strategy, version):
    capability = _scheduled(request, strategy, version)
    if not capability.candidates:
        return capability
    return Capability(tuple(replace(candidate, arithmetic=arithmetic,
                                   cost=Cost(CONSTANT_MATRIX_COST_PROFILE.cycles(arithmetic, candidate.input.words),
                                             candidate.cost.resource, candidate.cost.latency),
                                   confidence="calibrated" if CONSTANT_MATRIX_COST_PROFILE.covers(
                                       request.semantic, arithmetic, request.context.part, request.context.clock_period,
                                       candidate.implementation) else "uncalibrated")
                            for candidate in capability.candidates
                            for arithmetic in request.context.arithmetic.get(request.semantic.convolution.id, ())),
                      capability.findings)


TEMPORAL_STRATEGIES = (
    StageStrategy("aria-wide-stream", 2, _temporal),
    StageStrategy("phara", 1, _temporal),
    StageStrategy("hls4ml-temporal-block", 1, _temporal),
    StageStrategy("aria-window-stream", 1, _scheduled),
    StageStrategy("aria-affine-window", 1, _affine),
    StageStrategy("identity-layout-view", 1, _layout),
    StageStrategy("aria-dense-wide", 1, _dense),
)
