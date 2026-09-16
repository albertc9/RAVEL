"""Lower an explicit captured-window schedule without making planning choices."""

SOURCE = r'''#ifndef RAVEL_WINDOWS_H_
#define RAVEL_WINDOWS_H_
#include "hls_stream.h"
#include "nnet_mult.h"
namespace ravel {
template<class IN, class OUT, class CONFIG, unsigned POSITIONS>
void scheduled_conv(hls::stream<IN>& input, hls::stream<OUT>& output,
                    typename CONFIG::weight_t weights[CONFIG::filt_height * CONFIG::n_chan * CONFIG::n_filt],
                    typename CONFIG::bias_t biases[CONFIG::n_filt]) {
    typedef typename IN::value_type scalar;
    typedef typename OUT::value_type result;
    typedef typename CONFIG::mult_config MULT;
    static_assert(CONFIG::filt_width == 1 && CONFIG::stride_width == 1, "Temporal windows required");
    static_assert(CONFIG::in_width % POSITIONS == 0, "Positions must divide the row");
    static_assert(IN::size == POSITIONS * CONFIG::n_chan, "Input lane contract");
    static_assert(OUT::size == POSITIONS * CONFIG::n_filt, "Output lane contract");
    // Each row slot belongs to one spatial position. Arithmetic reads a local
    // snapshot; history receives only the next input, never an arithmetic result.
    static scalar history[CONFIG::filt_height][CONFIG::in_width][CONFIG::n_chan];
    #pragma HLS ARRAY_PARTITION variable=history complete dim=0
    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
WindowWords:
    for (unsigned word = 0; word < CONFIG::in_height * CONFIG::in_width / POSITIONS; ++word) {
        #pragma HLS PIPELINE II=1
        const unsigned row = word / (CONFIG::in_width / POSITIONS);
        const unsigned column = (word % (CONFIG::in_width / POSITIONS)) * POSITIONS;
        IN incoming = input.read();
        scalar window[POSITIONS][CONFIG::filt_height * CONFIG::n_chan];
        #pragma HLS ARRAY_PARTITION variable=window complete dim=0
        for (unsigned position = 0; position < POSITIONS; ++position) {
            #pragma HLS UNROLL
            for (unsigned tap = 0; tap < CONFIG::filt_height; ++tap) {
                #pragma HLS UNROLL
                for (unsigned channel = 0; channel < CONFIG::n_chan; ++channel) {
                    #pragma HLS UNROLL
                    window[position][tap * CONFIG::n_chan + channel] =
                        tap + 1 == CONFIG::filt_height ? incoming[position * CONFIG::n_chan + channel]
                        : history[tap + 1][column + position][channel];
                    history[tap][column + position][channel] = window[position][tap * CONFIG::n_chan + channel];
                }
            }
        }
        if (row + 1 >= CONFIG::filt_height && (row + 1 - CONFIG::filt_height) % CONFIG::stride_height == 0) {
            OUT outgoing;
            #pragma HLS ARRAY_PARTITION variable=outgoing complete dim=0
            for (unsigned position = 0; position < POSITIONS; ++position) {
                #pragma HLS UNROLL
                typename MULT::accum_t products[MULT::n_in][MULT::n_out];
                typename MULT::accum_t sums[MULT::n_out];
                #pragma HLS ARRAY_PARTITION variable=products complete dim=0
                #pragma HLS ARRAY_PARTITION variable=sums complete dim=0
                for (unsigned i = 0; i < MULT::n_in; ++i) {
                    #pragma HLS UNROLL
                    for (unsigned f = 0; f < MULT::n_out; ++f) {
                        #pragma HLS UNROLL
                        products[i][f] = MULT::template product<scalar, typename MULT::weight_t>::product(window[position][i], weights[i * MULT::n_out + f]);
                    }
                }
                for (unsigned f = 0; f < MULT::n_out; ++f) {
                    #pragma HLS UNROLL
                    sums[f] = (typename MULT::accum_t)biases[f];
                }
                // Preserve native dense_latency's per-product cast and ordered
                // fixed-point accumulation, including wrap and rounding modes.
                for (unsigned i = 0; i < MULT::n_in; ++i) {
                    #pragma HLS UNROLL
                    for (unsigned f = 0; f < MULT::n_out; ++f) {
                        #pragma HLS UNROLL
                        sums[f] += products[i][f];
                    }
                }
                for (unsigned f = 0; f < MULT::n_out; ++f) {
                    #pragma HLS UNROLL
                    outgoing[position * CONFIG::n_filt + f] = nnet::cast<scalar, result, MULT>(sums[f]);
                }
            }
            output.write(outgoing);
        }
    }
}

template<class IN, class OUT, class CONFIG, unsigned POSITIONS>
void scheduled_pool(hls::stream<IN>& input, hls::stream<OUT>& output) {
    static_assert(CONFIG::pool_height == CONFIG::stride_height && CONFIG::pool_width == 1, "Nonoverlapping temporal pool required");
    static_assert(IN::size == POSITIONS * CONFIG::n_filt && OUT::size == IN::size, "Pool lane contract");
    typename CONFIG::accum_t maxima[CONFIG::in_width][CONFIG::n_filt];
    #pragma HLS ARRAY_PARTITION variable=maxima complete dim=0
PoolWords:
    for (unsigned word = 0; word < CONFIG::in_height * CONFIG::in_width / POSITIONS; ++word) {
        #pragma HLS PIPELINE II=1
        const unsigned row = word / (CONFIG::in_width / POSITIONS);
        const unsigned column = (word % (CONFIG::in_width / POSITIONS)) * POSITIONS;
        IN incoming = input.read();
        OUT outgoing;
        #pragma HLS ARRAY_PARTITION variable=outgoing complete dim=0
        for (unsigned position = 0; position < POSITIONS; ++position) {
            #pragma HLS UNROLL
            for (unsigned f = 0; f < CONFIG::n_filt; ++f) {
                #pragma HLS UNROLL
                // hls4ml max-pool compares values after the pool accumulator cast.
                typename CONFIG::accum_t value = incoming[position * CONFIG::n_filt + f];
                if (row % CONFIG::pool_height == 0 || value > maxima[column + position][f])
                    maxima[column + position][f] = value;
                outgoing[position * CONFIG::n_filt + f] = maxima[column + position][f];
            }
        }
        if ((row + 1) % CONFIG::pool_height == 0) output.write(outgoing);
    }
}
}
#endif
'''


def emit_window(stage, native, payload, current_symbol, current_type, stream, observe, calls, typedefs):
    positions = stage["implementation"]["positions"]
    convolution, activation, pooling = stage["operation_ids"]
    for operation, function in ((convolution, "ravel::scheduled_conv"),
                                (activation, "nnet::relu"), (pooling, "ravel::scheduled_pool")):
        binding = native[operation]
        output_type = f"ravel_{operation}_window_t"
        output_symbol = binding["output_symbol"]
        typedefs.append(f"typedef nnet::array<{binding['output_precision_cpp']}, {stage['output']['lanes']}> {output_type};")
        stream(output_type, output_symbol)
        config = binding["config_symbol"]
        if operation == activation:
            config = config.replace("config", "relu_config")
        arguments = f"{current_symbol}, {output_symbol}"
        template = f"{current_type}, {output_type}, {config}"
        if operation != activation:
            template += f", {positions}"
        if operation == convolution:
            arguments += f", {payload[f'{operation}:weight'].symbol}, {payload[f'{operation}:bias'].symbol}"
        calls.append(f"    {function}<{template}>({arguments});")
        if operation != pooling:
            observe(f"{operation}:out0", output_symbol, binding["output_shape"])
        current_symbol, current_type = output_symbol, output_type
    return current_symbol, current_type


from .affine import emit_affine

LOWERINGS = {"aria-window-stream": emit_window, "aria-affine-window": emit_affine}
