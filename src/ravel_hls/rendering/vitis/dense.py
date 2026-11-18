"""Lower a selected sequential Dense matrix schedule without making planning choices."""

import numpy as np

from ...exceptions import ProjectGenerationError
from ...packing import pack_fixed_point_words


HEADER = "firmware/nnet_utils/ravel_dense.h"

SOURCE = r'''#ifndef RAVEL_DENSE_H_
#define RAVEL_DENSE_H_
#include "ap_int.h"
#include "hls_stream.h"
#include "nnet_mult.h"
namespace ravel {
// One input group meets OUT_LANES outputs per step. The products of a step are
// reduced outside the accumulator loop, so the loop carries one addition. The
// native Dense adds the same products one by one in a wrapping accumulator,
// which gives the same codes.
template<class IN, class OUT, class CONFIG, unsigned MAC_LANES, unsigned OUT_LANES, unsigned WEIGHT_BITS, unsigned STEPS>
void dense_matrix_stream(hls::stream<IN>& input, hls::stream<OUT>& output,
                         const ap_uint<MAC_LANES * WEIGHT_BITS> packed_weights[OUT_LANES][STEPS],
                         typename CONFIG::bias_t biases[CONFIG::n_out]) {
    typedef typename IN::value_type scalar;
    typedef typename OUT::value_type result;
    static_assert(OUT::size == CONFIG::n_out, "Dense matrix output word must hold every output");
    static_assert(CONFIG::n_in % IN::size == 0, "Dense matrix input packing mismatch");
    static_assert(IN::size % MAC_LANES == 0, "Dense matrix lanes must divide the input word");
    static_assert(CONFIG::n_out % OUT_LANES == 0, "Dense matrix output lanes must divide the outputs");
    constexpr unsigned GROUPS = IN::size / MAC_LANES;
    constexpr unsigned PHASES = CONFIG::n_out / OUT_LANES;
    static_assert(STEPS == CONFIG::n_in / MAC_LANES * PHASES, "Dense matrix depth mismatch");
    typename CONFIG::accum_t accumulators[CONFIG::n_out];
    #pragma HLS ARRAY_PARTITION variable=accumulators complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    for (unsigned unit = 0; unit < CONFIG::n_out; ++unit) {
        #pragma HLS UNROLL
        accumulators[unit] = (typename CONFIG::accum_t)biases[unit];
    }
    IN input_word;
    unsigned group = 0, phase = 0;
DenseMatrix:
    for (unsigned step = 0; step < STEPS; ++step) {
        #pragma HLS PIPELINE II=1
        if (group == 0 && phase == 0) input_word = input.read();
        for (unsigned out = 0; out < OUT_LANES; ++out) {
            #pragma HLS UNROLL
            const ap_uint<MAC_LANES * WEIGHT_BITS> packed = packed_weights[out][step];
            typename CONFIG::accum_t products[MAC_LANES];
            #pragma HLS ARRAY_PARTITION variable=products complete
            #pragma HLS BIND_OP variable=products op=mul impl=dsp
            for (unsigned lane = 0; lane < MAC_LANES; ++lane) {
                #pragma HLS UNROLL
                typename CONFIG::weight_t weight = 0;
                weight.range(WEIGHT_BITS - 1, 0) = packed.range((lane + 1) * WEIGHT_BITS - 1, lane * WEIGHT_BITS);
                products[lane] = (typename CONFIG::accum_t)CONFIG::template product<
                    scalar, typename CONFIG::weight_t>::product(input_word[group * MAC_LANES + lane], weight);
            }
            typename CONFIG::accum_t partial = 0;
            for (unsigned lane = 0; lane < MAC_LANES; ++lane) {
                #pragma HLS UNROLL
                partial += products[lane];
            }
            accumulators[phase * OUT_LANES + out] += partial;
        }
        if (phase + 1 == PHASES) {
            phase = 0;
            group = group + 1 == GROUPS ? 0 : group + 1;
        } else {
            ++phase;
        }
    }
    OUT outgoing;
    for (unsigned unit = 0; unit < CONFIG::n_out; ++unit) {
        #pragma HLS UNROLL
        outgoing[unit] = nnet::cast<scalar, result, CONFIG>(accumulators[unit]);
    }
    output.write(outgoing);
}
}
#endif
'''


def matrix_words(parameter, schedule):
    """One ROM per output lane; a word holds one input group for the output of that lane and step.

    Steps run over input groups and, inside a group, over output phases.
    """
    inputs, outputs = schedule["inputs"], schedule["outputs"]
    mac_lanes, output_lanes = schedule["mac_lanes"], schedule["output_lanes"]
    values = np.asarray(parameter.values).reshape(inputs, outputs)
    precision = parameter.numeric_type
    roms = []
    for lane in range(output_lanes):
        ordered = [value
                   for group in range(inputs // mac_lanes)
                   for phase in range(outputs // output_lanes)
                   for value in values[group * mac_lanes:(group + 1) * mac_lanes, phase * output_lanes + lane]]
        try:
            words = pack_fixed_point_words(ordered, width=precision["width"], integer=precision["integer"],
                                           signed=precision["signed"], lanes=mac_lanes)
        except ValueError as error:
            raise ProjectGenerationError(f"Dense weight {parameter.symbol} cannot be packed: {error}") from error
        if len(words) != schedule["steps"]:
            raise ProjectGenerationError(f"Dense weight {parameter.symbol} packed depth does not match its schedule")
        roms.append(words)
    return roms


def weights_path(parameter):
    return f"firmware/weights/{parameter.symbol}_ravel_matrix.h"


def weights_header(parameter, schedule):
    roms = matrix_words(parameter, schedule)
    bits = schedule["mac_lanes"] * parameter.numeric_type["width"]
    digits = (bits + 3) // 4
    name = f"{parameter.symbol}_ravel_matrix"
    guard = f"RAVEL_{parameter.symbol.upper()}_MATRIX_H_"
    lines = [f"#ifndef {guard}", f"#define {guard}", "", '#include "ap_int.h"', "",
             f"const ap_uint<{bits}> {name}[{len(roms)}][{schedule['steps']}] = {{"]
    for lane, words in enumerate(roms):
        lines.append("    {")
        lines += [f'        ap_uint<{bits}>("0x{word:0{digits}x}", 16)' + ("," if index + 1 < len(words) else "")
                  for index, word in enumerate(words)]
        lines.append("    }" + ("," if lane + 1 < len(roms) else ""))
    lines += ["};", "", "#endif", ""]
    return "\n".join(lines)


def emit_matrix(stage, native, payload, current_symbol, current_type, stream):
    """Return the call of one Dense matrix stage and its output stream."""
    operation = stage["operation_ids"][0]
    binding = native[operation]
    schedule = stage["dense"]
    weight, bias = payload[f"{operation}:weight"], payload[f"{operation}:bias"]
    stream(binding["output_type"], binding["output_symbol"])
    arguments = (f"{current_type}, {binding['output_type']}, {binding['config_symbol']}, {schedule['mac_lanes']}, "
                 f"{schedule['output_lanes']}, {weight.numeric_type['width']}, {schedule['steps']}")
    call = (f"    ravel::dense_matrix_stream<{arguments}>({current_symbol}, {binding['output_symbol']}, "
            f"{weight.symbol}_ravel_matrix, {bias.symbol});")
    return call, binding["output_symbol"], binding["output_type"]
