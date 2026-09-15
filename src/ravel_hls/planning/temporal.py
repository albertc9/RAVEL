"""Qualified temporal-stage candidates composed by the generic chain solver."""

from math import prod

from ..domain.temporal import TemporalChain
from .chain import Candidate, ChainPlan, Cost, StreamContract, resolve_chain


def plan_temporal_chain(chain: TemporalChain, *, temporal_packing: int,
                        dense_parallelism: int, input_strategy: str,
                        input_cycles: int, dense_cycles: int) -> ChainPlan:
    """Compose the qualified packed-input region, native blocks and Dense.

    The existing temporal realization owns the packed external input contract.
    Native variants own internal channel-mixing endpoints. This qualification
    restriction is about endpoint contracts, independent of source layer IDs.
    """
    domains = []
    external_endpoint = chain.blocks[0].input.id
    for block in chain.blocks:
        convolution = block.convolution
        pooled = block.pooling.outputs[0]
        input_lanes = int(convolution.attribute("n_chan"))
        output_lanes = int(convolution.attribute("n_filt"))
        external = block.input.id == external_endpoint
        if external:
            input_lanes *= block.input.shape[1] * temporal_packing
            output_lanes *= pooled.shape[1]
        source = StreamContract(block.input.id, block.input.shape, block.input.numeric_type, input_lanes)
        target = StreamContract(pooled.id, pooled.shape, pooled.numeric_type, output_lanes)
        multipliers = prod(parameter.shape[0:-1]) * parameter.shape[-1] if (parameter := next((p for p in convolution.parameters if p.role == "weight"), None)) else 0
        native_cycles = max(source.words, target.words)
        candidate = Candidate(input_strategy if external else "hls4ml-temporal-block", 1,
                              (convolution.id, block.activation.id, block.pooling.id), source, target,
                              Cost(input_cycles if external else native_cycles, multipliers, native_cycles), external,
                              "analytical" if external else "uncalibrated-native")
        domains.append((candidate,))
    layout = chain.layout
    previous = domains[-1][0].output
    flattened = StreamContract(layout.output.id, layout.output.shape, layout.output.numeric_type, previous.lanes)
    domains.append((Candidate("identity-layout-view", 1, (layout.operation.id,), previous, flattened, Cost(0, 0, 0), False),))
    dense_lanes = layout.input.shape[-1] * layout.input.shape[-2]
    dense_input = StreamContract(layout.output.id, layout.output.shape, layout.output.numeric_type, dense_lanes)
    head_output = chain.head.outputs[0]
    dense_output = StreamContract(head_output.id, head_output.shape, head_output.numeric_type, 1)
    domains.append((Candidate("aria-dense-wide", 1, (chain.head.id,), dense_input, dense_output,
                             Cost(dense_cycles, layout.input.shape[-1] * dense_parallelism, dense_cycles), True),))
    return resolve_chain(tuple(domains))
