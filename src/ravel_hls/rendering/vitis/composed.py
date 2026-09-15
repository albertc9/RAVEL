"""Compose selected native calls, RAVEL stages and explicit stream bridges."""

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from ...domain import ParameterPayload
from ...exceptions import ProjectGenerationError
from .renderer import render_aria_project


def render_project(path: Path, name: str, design: Mapping[str, Any], parameters: ParameterPayload) -> list[str]:
    if design["strategy"]["id"] != "aria-composed":
        return render_aria_project(path, name, design, parameters)
    managed = render_aria_project(path, name, design, parameters)
    rendering = design["rendering"]
    native = rendering["native_operations"]
    legacy = rendering["operations"]
    plan = design["implementation_plan"]
    payload = parameters.by_id()
    input_binding = native[design["stages"][0]["input"]["tensor_id"].split(":")[0]]
    output_binding = native[design["stages"][-1]["operation_ids"][-1]]
    current_symbol = input_binding["output_symbol"]
    current_type = rendering["types"]["input_wide"]
    current_contract = design["stages"][0]["input"]
    declarations = []
    calls = []
    typedefs = []
    bridge_index = 0

    def stream(type_name, symbol):
        declarations.extend([f'    hls::stream<{type_name}> {symbol}("{symbol}");',
                             f'    #pragma HLS STREAM variable={symbol} depth=4'])

    for stage in design["stages"]:
        if current_contract != stage["input"]:
            bridge = next((item for item in design["bridges"] if item["input"] == current_contract and item["output"] == stage["input"]), None)
            if bridge is None:
                raise ProjectGenerationError("Composed stage has no resolved endpoint bridge")
            target = bridge["output"]
            binding = native[target["tensor_id"].split(":")[0]]
            bridge_type = f"ravel_bridge_{bridge_index}_t"
            symbol = f"ravel_bridge_{bridge_index}"
            typedefs.append(f"typedef nnet::array<{binding['output_precision_cpp']}, {target['lanes']}> {bridge_type};")
            stream(bridge_type, symbol)
            count = 1
            for extent in target["shape"]:
                count *= extent
            calls.append(f"    ravel::repack<{current_type}, {bridge_type}, {count}>({current_symbol}, {symbol});")
            current_symbol, current_type = symbol, bridge_type
            bridge_index += 1
        strategy = stage["strategy"]["id"]
        if strategy in {"aria-wide-stream", "phara"}:
            convolution, activation, pooling = stage["operation_ids"]
            conv = legacy[convolution]
            relu = legacy[activation]
            pool = legacy[pooling]
            conv_type = rendering["types"]["convolution_wide"]
            relu_type = rendering["types"]["activation_wide"]
            pool_type = rendering["types"]["pooling_wide"]
            conv_symbol = rendering["streams"]["convolution"]
            relu_symbol = rendering["streams"]["activation"]
            pool_symbol = rendering["streams"]["pooling"]
            stream(pool_type, pool_symbol)
            weight, bias = payload[f"{convolution}:weight"], payload[f"{convolution}:bias"]
            if strategy == "phara":
                calls.append(f"    nnet::{rendering['phara_fused_function']}<{current_type}, {conv_type}, {relu_type}, {pool_type}, {conv['config_symbol']}, {pool['config_symbol']}>({current_symbol}, {pool_symbol}, {weight.symbol}, {bias.symbol});")
            else:
                stream(conv_type, conv_symbol)
                stream(relu_type, relu_symbol)
                calls.extend([
                    f"    nnet::{rendering['first_convolution_function']}<{current_type}, {conv_type}, {conv['config_symbol']}>({current_symbol}, {conv_symbol}, {weight.symbol}, {bias.symbol});",
                    f"    nnet::relu<{conv_type}, {relu_type}, {relu['config_symbol']}>({conv_symbol}, {relu_symbol});",
                    f"    nnet::maxpool2d_wide_nonoverlap_cl<{relu_type}, {pool_type}, {pool['config_symbol']}>({relu_symbol}, {pool_symbol});",
                ])
            current_symbol, current_type = pool_symbol, pool_type
        elif strategy == "hls4ml-temporal-block":
            for operation_id in stage["operation_ids"]:
                binding = native[operation_id]
                stream(binding["output_type"], binding["output_symbol"])
                native_call = binding["native_call"]
                if not isinstance(native_call, str):
                    raise ProjectGenerationError("Delegated operation has no qualified native call")
                native_call = re.sub(r"\b" + re.escape(binding["input_symbol"]) + r"\b", current_symbol, native_call)
                calls.append("    " + native_call)
                current_symbol, current_type = binding["output_symbol"], binding["output_type"]
        elif strategy == "identity-layout-view":
            pass
        elif strategy == "aria-dense-wide":
            operation_id = stage["operation_ids"][0]
            binding = native[operation_id]
            weight, bias = payload[f"{operation_id}:weight"], payload[f"{operation_id}:bias"]
            calls.append(f"    nnet::dense_wide_stream<{current_type}, {binding['output_type']}, {binding['config_symbol']}>({current_symbol}, {binding['output_symbol']}, {weight.symbol}_ravel_packed, {bias.symbol});")
            current_symbol, current_type = binding["output_symbol"], binding["output_type"]
        else:
            raise ProjectGenerationError(f"Unknown composed strategy: {strategy}")
        current_contract = stage["output"]

    dense_weight = payload[f"{design['stages'][-1]['operation_ids'][0]}:weight"]
    loads = [f'        nnet::load_weights_from_txt<{tensor.type_name}, {tensor.values.size}>({tensor.symbol}, "{tensor.symbol}.txt");'
             for tensor in parameters.tensors]
    text = '\n'.join([
        f'#include "{name}.h"', '#include "parameters.h"', '#include "nnet_utils/nnet_aria.h"',
        '#include "nnet_utils/ravel_bridges.h"', f'#include "weights/{dense_weight.symbol}_ravel_packed.h"',
        f'void {name}(hls::stream<{rendering["types"]["input_wide"]}> &{input_binding["output_symbol"]}, hls::stream<{output_binding["output_type"]}> &{output_binding["output_symbol"]}) {{',
        f'    #pragma HLS INTERFACE axis port={input_binding["output_symbol"]},{output_binding["output_symbol"]}',
        '    #pragma HLS DATAFLOW', '#ifndef __SYNTHESIS__', '    static bool loaded = false;',
        '    if (!loaded) {', *loads, '        loaded = true;', '    }', '#endif',
        *declarations, *calls, '}', '',
    ])
    (path / f"firmware/{name}.cpp").write_text(text)
    defines = path / "firmware/defines.h"
    body, marker, suffix = defines.read_text().rpartition("#endif")
    if not marker:
        raise ProjectGenerationError("Composed defines have no include guard")
    defines.write_text(body + "\n".join(typedefs) + "\n#endif" + suffix)
    bridge_header = "firmware/nnet_utils/ravel_bridges.h"
    (path / bridge_header).write_text(BRIDGE_SOURCE)
    return sorted({*managed, bridge_header})


BRIDGE_SOURCE = '''#ifndef RAVEL_BRIDGES_H_
#define RAVEL_BRIDGES_H_
#include "hls_stream.h"
namespace ravel {
template<class IN, class OUT, unsigned COUNT>
void repack(hls::stream<IN>& input, hls::stream<OUT>& output) {
    typedef typename IN::value_type input_value;
    typedef typename OUT::value_type output_value;
    static_assert(input_value::width == output_value::width, "Bridge must preserve scalar bits");
    IN incoming;
    OUT outgoing;
BridgeCodes:
    for (unsigned i = 0; i < COUNT; ++i) {
        #pragma HLS PIPELINE II=1
        if (i % IN::size == 0) incoming = input.read();
        if (i % OUT::size == 0) {
            for (unsigned lane = 0; lane < OUT::size; ++lane) {
                #pragma HLS UNROLL
                outgoing[lane] = 0;
            }
        }
        outgoing[i % OUT::size].range(input_value::width - 1, 0) = incoming[i % IN::size].range(input_value::width - 1, 0);
        if (i % OUT::size == OUT::size - 1 || i == COUNT - 1) output.write(outgoing);
    }
}
}
#endif
'''
