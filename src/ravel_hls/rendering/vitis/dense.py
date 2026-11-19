"""Lower a selected sequential Dense matrix schedule without making planning choices."""

import numpy as np

from ...exceptions import ProjectGenerationError
from ...packing import pack_fixed_point_words


HEADER = "firmware/nnet_utils/ravel_dense.h"

PROLOGUE = r'''#ifndef RAVEL_DENSE_H_
#define RAVEL_DENSE_H_
#include "ap_int.h"
#include "hls_stream.h"
#include "nnet_mult.h"
namespace ravel {
'''

EPILOGUE = r'''}
#endif
'''

ENGINE = r'''// One input group meets OUT_LANES outputs per step. The products of a step are
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
'''

SHARED_ENGINE = r'''// Two outputs of a step meet the same input values, so one multiplier forms
// both products of an input value: (high * 2^FIELD + low) * x. FIELD is the
// signed width of one product, so the low field is low * x exactly, and
// high * x is the remaining field plus the sign bit of the low field. Each
// product then takes the unchanged product type and accumulator cast.
// X_BITS and W_BITS are the signed widths of the input and weight codes.
template<class IN, class OUT, class CONFIG, unsigned MAC_LANES, unsigned OUT_LANES, unsigned WEIGHT_BITS, unsigned STEPS,
         unsigned X_BITS, unsigned W_BITS>
void dense_matrix_shared_stream(hls::stream<IN>& input, hls::stream<OUT>& output,
                                const ap_uint<MAC_LANES * WEIGHT_BITS> packed_weights[OUT_LANES][STEPS],
                                typename CONFIG::bias_t biases[CONFIG::n_out]) {
    typedef typename IN::value_type scalar;
    typedef typename OUT::value_type result;
    typedef typename CONFIG::weight_t weight_t;
    typedef decltype(CONFIG::template product<scalar, weight_t>::product(scalar(), weight_t())) product_t;
    static_assert(OUT::size == CONFIG::n_out, "Dense matrix output word must hold every output");
    static_assert(CONFIG::n_in % IN::size == 0, "Dense matrix input packing mismatch");
    static_assert(IN::size % MAC_LANES == 0, "Dense matrix lanes must divide the input word");
    static_assert(CONFIG::n_out % OUT_LANES == 0, "Dense matrix output lanes must divide the outputs");
    static_assert(OUT_LANES % 2 == 0, "Shared multipliers need the outputs of a step in pairs");
    constexpr unsigned GROUPS = IN::size / MAC_LANES;
    constexpr unsigned PHASES = CONFIG::n_out / OUT_LANES;
    constexpr unsigned FIELD = X_BITS + W_BITS;
    constexpr unsigned OPERAND = FIELD + W_BITS + 1;
    constexpr unsigned X_WIDTH = scalar::width;
    constexpr unsigned PRODUCT = product_t::width;
    static_assert(PRODUCT <= FIELD, "Shared multiplier field must hold one product");
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
        for (unsigned pair = 0; pair < OUT_LANES / 2; ++pair) {
            #pragma HLS UNROLL
            const ap_uint<MAC_LANES * WEIGHT_BITS> low_word = packed_weights[2 * pair][step];
            const ap_uint<MAC_LANES * WEIGHT_BITS> high_word = packed_weights[2 * pair + 1][step];
            ap_int<OPERAND + X_BITS> wide[MAC_LANES];
            #pragma HLS ARRAY_PARTITION variable=wide complete
            #pragma HLS BIND_OP variable=wide op=mul impl=dsp
            typename CONFIG::accum_t low_products[MAC_LANES], high_products[MAC_LANES];
            #pragma HLS ARRAY_PARTITION variable=low_products complete
            #pragma HLS ARRAY_PARTITION variable=high_products complete
            for (unsigned lane = 0; lane < MAC_LANES; ++lane) {
                #pragma HLS UNROLL
                const scalar value = input_word[group * MAC_LANES + lane];
                const ap_uint<X_WIDTH> value_bits = value.range(X_WIDTH - 1, 0);
                const ap_uint<WEIGHT_BITS> low_bits = low_word.range((lane + 1) * WEIGHT_BITS - 1, lane * WEIGHT_BITS);
                const ap_uint<WEIGHT_BITS> high_bits = high_word.range((lane + 1) * WEIGHT_BITS - 1, lane * WEIGHT_BITS);
                const ap_int<X_BITS> x = value_bits;
                const ap_int<W_BITS> low_weight = low_bits;
                const ap_int<W_BITS> high_weight = high_bits;
                const ap_int<OPERAND> operand = (((ap_int<OPERAND>)high_weight) << FIELD) + low_weight;
                wide[lane] = operand * x;
                const ap_uint<FIELD> low_field = wide[lane].range(FIELD - 1, 0);
                const ap_int<FIELD> low = low_field;
                const ap_int<FIELD> high = (wide[lane] >> FIELD) + wide[lane][FIELD - 1];
                product_t low_product = 0, high_product = 0;
                low_product.range(PRODUCT - 1, 0) = low.range(PRODUCT - 1, 0);
                high_product.range(PRODUCT - 1, 0) = high.range(PRODUCT - 1, 0);
                low_products[lane] = (typename CONFIG::accum_t)low_product;
                high_products[lane] = (typename CONFIG::accum_t)high_product;
            }
            typename CONFIG::accum_t low_partial = 0, high_partial = 0;
            for (unsigned lane = 0; lane < MAC_LANES; ++lane) {
                #pragma HLS UNROLL
                low_partial += low_products[lane];
                high_partial += high_products[lane];
            }
            accumulators[phase * OUT_LANES + 2 * pair] += low_partial;
            accumulators[phase * OUT_LANES + 2 * pair + 1] += high_partial;
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
'''


def source(schedules):
    """Header text with only the engines the recorded schedules use."""
    shared = [bool(schedule.get("packing")) for schedule in schedules]
    return PROLOGUE + (ENGINE if not all(shared) else "") + (SHARED_ENGINE if any(shared) else "") + EPILOGUE


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
    function = "dense_matrix_stream"
    if schedule.get("packing"):
        # The field is the sum of the signed operand widths; the weight's follows from its recorded type.
        weight_bits = weight.numeric_type["width"] + (0 if weight.numeric_type["signed"] else 1)
        arguments += f", {schedule['packing']['field_bits'] - weight_bits}, {weight_bits}"
        function = "dense_matrix_shared_stream"
    call = (f"    ravel::{function}<{arguments}>({current_symbol}, {binding['output_symbol']}, "
            f"{weight.symbol}_ravel_matrix, {bias.symbol});")
    return call, binding["output_symbol"], binding["output_type"]
