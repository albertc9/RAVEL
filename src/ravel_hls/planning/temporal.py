"""Enumerate typed capabilities and resolve the complete temporal chain."""

from ..domain.temporal import TemporalChain, Finding
from .chain import ChainPlan, TEMPORAL_RESOLVER
from .bridges import LOSSLESS_BRIDGES
from .strategies import StageContext, StageRequest, TEMPORAL_STRATEGIES
from .search import SearchReport


def plan_temporal_chain(chain: TemporalChain, *, temporal_packing: int,
                        dense_parallelism: int, input_strategy: str,
                        input_cycles: int, dense_cycles: int,
                        strategies=TEMPORAL_STRATEGIES, bridges=LOSSLESS_BRIDGES, resolver=TEMPORAL_RESOLVER) -> SearchReport:
    context = StageContext(chain.blocks[0].input.id, input_strategy, temporal_packing,
                           dense_parallelism, input_cycles, dense_cycles)
    findings = []
    def candidates(request):
        evaluations = [strategy.evaluate(request) for strategy in sorted(strategies, key=lambda entry: (entry.id, entry.version))]
        result = tuple(candidate for evaluation in evaluations for candidate in evaluation.candidates)
        if not result:
            findings.extend(finding for evaluation in evaluations for finding in evaluation.findings)
        return result
    domains = []
    for block in chain.blocks:
        domain = candidates(StageRequest(block, context))
        if not domain:
            failed = ChainPlan(findings=tuple(findings) or (Finding("planner.no_stage_candidate", "No qualified strategy can implement this temporal block", block.convolution.id),))
            return SearchReport((), failed)
        domains.append(domain)
    plans = []
    for downstream in domains[-1]:
        layouts = candidates(StageRequest(chain.layout, context, previous=downstream.output))
        head = candidates(StageRequest(chain.head, context, layout=chain.layout))
        domain = (*domains[:-1], (downstream,), layouts, head)
        plan = resolver.resolve(domain, bridge_strategies=bridges)
        if not plan.findings:
            plans.append(plan)
    incumbent = next((plan for plan in plans if not any(stage.implementation for stage in plan.stages)), None)
    selected = incumbent or next(iter(plans), ChainPlan(findings=(Finding("planner.no_qualified_plan", "No compatible temporal implementation"),)))
    return SearchReport(tuple(plans), selected)
