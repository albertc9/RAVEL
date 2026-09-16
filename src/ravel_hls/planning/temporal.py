"""Enumerate typed capabilities and resolve the complete temporal chain."""

from ..domain.temporal import TemporalChain, Finding
from .chain import ChainPlan, TEMPORAL_RESOLVER
from .bridges import LOSSLESS_BRIDGES
from .strategies import StageContext, StageRequest, TEMPORAL_STRATEGIES


def plan_temporal_chain(chain: TemporalChain, *, temporal_packing: int,
                        dense_parallelism: int, input_strategy: str,
                        input_cycles: int, dense_cycles: int,
                        strategies=TEMPORAL_STRATEGIES, bridges=LOSSLESS_BRIDGES, resolver=TEMPORAL_RESOLVER) -> ChainPlan:
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
            return ChainPlan(findings=tuple(findings) or (Finding("planner.no_stage_candidate", "No qualified strategy can implement this temporal block", block.convolution.id),))
        domains.append(domain)
    layouts = tuple(candidate for preceding in domains[-1]
                    for candidate in candidates(StageRequest(chain.layout, context, previous=preceding.output)))
    domains.append(layouts)
    domains.append(candidates(StageRequest(chain.head, context, layout=chain.layout)))
    return resolver.resolve(tuple(domains), bridge_strategies=bridges)
