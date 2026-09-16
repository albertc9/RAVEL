"""Lower a proven matrix without choosing arithmetic or scheduling policies."""

import re

from ...exceptions import ProjectGenerationError


_EXPRESSIONS = {
    "constant": lambda operands, value: str(value),
    "multiply": lambda operands, value: f"{operands[0]} * ({value})",
    "shift": lambda operands, value: f"{operands[0]} << {value}",
    "add": lambda operands, value: f"{operands[0]} + {operands[1]}",
    "subtract": lambda operands, value: f"{operands[0]} - {operands[1]}",
    "negate": lambda operands, value: f"-{operands[0]}",
}


def matrix_struct(arithmetic, symbol):
    if arithmetic["proof"]["status"] != "proven":
        raise ProjectGenerationError("Constant matrix requires an exact modular proof")
    graph = arithmetic["graph"]
    width = arithmetic["accumulator_numeric"]["width"]
    source = arithmetic["input_numeric"]
    code_type = "ap_int" if source["signed"] else "ap_uint"
    ids = {name: re.sub(r"\W", "_", name) for name in (
        *graph["input_ids"], *(node["id"] for node in graph["nodes"]))}
    lines = [f"struct {symbol} {{", "template<class SCALAR, class MULT>",
             "static void apply(SCALAR *input, typename MULT::accum_t *output) {",
             "#pragma HLS INLINE", f"typedef ap_int<{width}> code_t;"]
    for index, name in enumerate(graph["input_ids"]):
        lines.extend([f"{code_type}<{source['width']}> {ids[name]}_raw;",
                      f"{ids[name]}_raw.range({source['width'] - 1}, 0) = input[{index}].range({source['width'] - 1}, 0);",
                      f"code_t {ids[name]} = {ids[name]}_raw;"])
    for node in graph["nodes"]:
        operands = [ids[name] for name in node["inputs"]]
        expression = _EXPRESSIONS[node["operation"]](operands, node["value"])
        lines.append(f"code_t {ids[node['id']]} = {expression};")
        if node["operation"] == "multiply":
            lines.append(f"#pragma HLS BIND_OP variable={ids[node['id']]} op=mul impl=dsp")
    for index, name in enumerate(graph["output_ids"]):
        lines.append(f"output[{index}].range({width - 1}, 0) = {ids[name]}.range({width - 1}, 0);")
    return "\n".join([*lines, "}", "};"])


SOURCE = r'''
namespace ravel {
template<class IN, class OUT, class CONFIG, unsigned POSITIONS, class ARITHMETIC>
void scheduled_affine_conv(hls::stream<IN>& input, hls::stream<OUT>& output) {
    typedef typename IN::value_type scalar;
    typedef typename OUT::value_type result;
    typedef typename CONFIG::mult_config MULT;
    static scalar history[CONFIG::filt_height][CONFIG::in_width][CONFIG::n_chan];
    #pragma HLS ARRAY_PARTITION variable=history complete dim=0
    unsigned row=0, column=0, phase=0;
AffineWords:
    for (unsigned word=0; word<CONFIG::in_height*CONFIG::in_width/POSITIONS; ++word) {
        #pragma HLS PIPELINE II=1
        IN incoming=input.read();
        scalar window[POSITIONS][CONFIG::filt_height*CONFIG::n_chan];
        #pragma HLS ARRAY_PARTITION variable=window complete dim=0
        for(unsigned position=0; position<POSITIONS; ++position) {
            #pragma HLS UNROLL
            for(unsigned tap=0; tap<CONFIG::filt_height; ++tap) {
                #pragma HLS UNROLL
                for(unsigned channel=0; channel<CONFIG::n_chan; ++channel) {
                    #pragma HLS UNROLL
                    window[position][tap*CONFIG::n_chan+channel] =
                        tap+1==CONFIG::filt_height ? incoming[position*CONFIG::n_chan+channel]
                        : history[tap+1][column+position][channel];
                    history[tap][column+position][channel]=window[position][tap*CONFIG::n_chan+channel];
                }
            }
        }
        if(row+1>=CONFIG::filt_height && phase==0) {
            OUT outgoing;
            #pragma HLS ARRAY_PARTITION variable=outgoing complete dim=0
            for(unsigned position=0; position<POSITIONS; ++position) {
                #pragma HLS UNROLL
                typename MULT::accum_t sums[MULT::n_out];
                #pragma HLS ARRAY_PARTITION variable=sums complete dim=0
                ARITHMETIC::template apply<scalar, MULT>(window[position], sums);
                for(unsigned f=0; f<MULT::n_out; ++f) {
                    #pragma HLS UNROLL
                    outgoing[position*CONFIG::n_filt+f]=nnet::cast<scalar,result,MULT>(sums[f]);
                }
            }
            output.write(outgoing);
        }
        column+=POSITIONS;
        if(column==CONFIG::in_width) {
            column=0;
            ++row;
            if(row>=CONFIG::filt_height)
                phase=phase+1==CONFIG::stride_height ? 0 : phase+1;
        }
    }
}
}
'''


def emit_affine(stage, native, payload, current_symbol, current_type, stream, observe, calls, typedefs):
    positions = stage["implementation"]["positions"]
    for operation in stage["operation_ids"]:
        binding = native[operation]
        output_type, output_symbol = f"ravel_{operation}_window_t", binding["output_symbol"]
        typedefs.append(f"typedef nnet::array<{binding['output_precision_cpp']}, {stage['output']['lanes']}> {output_type};")
        stream(output_type, output_symbol)
        if operation == stage["operation_ids"][0]:
            arguments = f"{current_type}, {output_type}, {binding['config_symbol']}, {positions}, ravel_matrix_{operation}"
            calls.append(f"    ravel::scheduled_affine_conv<{arguments}>({current_symbol}, {output_symbol});")
        elif operation == stage["operation_ids"][1]:
            config = binding["config_symbol"].replace("config", "relu_config")
            calls.append(f"    nnet::relu<{current_type}, {output_type}, {config}>({current_symbol}, {output_symbol});")
        else:
            calls.append(f"    ravel::scheduled_pool<{current_type}, {output_type}, {binding['config_symbol']}, {positions}>({current_symbol}, {output_symbol});")
        if operation != stage["operation_ids"][-1]:
            observe(f"{operation}:out0", output_symbol, binding["output_shape"])
        current_symbol, current_type = output_symbol, output_type
    return current_symbol, current_type
