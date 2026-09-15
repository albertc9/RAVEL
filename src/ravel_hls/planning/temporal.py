"""Enumerate typed capabilities and resolve the complete temporal chain."""

from ..domain.temporal import TemporalChain
from .chain import ChainPlan, resolve_chain
from .strategies import StageContext, StageRequest, TEMPORAL_STRATEGIES


def plan_temporal_chain(chain: TemporalChain, *, temporal_packing: int,
                        dense_parallelism: int, input_strategy: str,
                        input_cycles: int, dense_cycles: int,
                        strategies=TEMPORAL_STRATEGIES) -> ChainPlan:
    context = StageContext(chain.blocks[0].input.id, input_strategy, temporal_packing,
                           dense_parallelism, input_cycles, dense_cycles)
    def candidates(request):
        return tuple(candidate for strategy in sorted(strategies, key=lambda entry: (entry.id, entry.version))
                     for candidate in strategy.evaluate(request).candidates)
    domains = [candidates(StageRequest(block, context)) for block in chain.blocks]
    layouts = tuple(candidate for preceding in domains[-1]
                    for candidate in candidates(StageRequest(chain.layout, context, previous=preceding.output)))
    domains.append(layouts)
    domains.append(candidates(StageRequest(chain.head, context, layout=chain.layout)))
    return resolve_chain(tuple(domains))
