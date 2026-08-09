#ifndef NNET_ARIA_H_
#define NNET_ARIA_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_mult.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void first_conv_2row_4lane_temporal_wide_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[
        CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    static_assert(CONFIG_T::n_chan == 1, "Aria first convolution requires one channel");
    static_assert(CONFIG_T::filt_width == 1, "Aria first convolution requires filt_width=1");
    static_assert(CONFIG_T::stride_width == 1, "Aria first convolution requires stride_width=1");
    static_assert(CONFIG_T::in_width * 2 == data_T::size, "Aria input word must contain two rows");
    static_assert(CONFIG_T::n_filt * CONFIG_T::out_width == res_T::size,
                  "Aria convolution output must contain every width/filter lane");
    static_assert(CONFIG_T::in_height % 2 == 0, "Aria input height must be even");
    static_assert(CONFIG_T::stride_height >= 2, "Aria pair schedule requires stride_height>=2");

    typedef typename data_T::value_type data_value_t;
    typedef typename res_T::value_type res_value_t;
    data_value_t row_buf[CONFIG_T::filt_height][CONFIG_T::in_width];
    #pragma HLS ARRAY_PARTITION variable=row_buf complete dim=0

    constexpr unsigned EMIT_ROW0_PHASE =
        ((CONFIG_T::filt_height - 1) / 2) % CONFIG_T::stride_height;
    constexpr unsigned EMIT_ROW1_PHASE =
        ((CONFIG_T::filt_height - 1 + CONFIG_T::stride_height) / 2) % CONFIG_T::stride_height;
    constexpr unsigned PAIR_WINDOW_START = CONFIG_T::filt_height / 2;
    unsigned pair_phase = 0;

ReadPairsWide:
    for (unsigned i_pair = 0; i_pair < CONFIG_T::in_height / 2; i_pair++) {
        #pragma HLS PIPELINE II=1
        data_T in_pack = data.read();
        data_value_t row0[CONFIG_T::in_width];
        data_value_t row1[CONFIG_T::in_width];
        #pragma HLS ARRAY_PARTITION variable=row0 complete
        #pragma HLS ARRAY_PARTITION variable=row1 complete
        for (unsigned c = 0; c < CONFIG_T::in_width; c++) {
            #pragma HLS UNROLL
            row0[c] = in_pack[c];
            row1[c] = in_pack[CONFIG_T::in_width + c];
        }

        data_value_t old_buf[CONFIG_T::filt_height][CONFIG_T::in_width];
        #pragma HLS ARRAY_PARTITION variable=old_buf complete dim=0
        for (unsigned b = 0; b < CONFIG_T::filt_height; b++) {
            #pragma HLS UNROLL
            for (unsigned c = 0; c < CONFIG_T::in_width; c++) {
                #pragma HLS UNROLL
                old_buf[b][c] = row_buf[b][c];
            }
        }
        for (unsigned c = 0; c < CONFIG_T::in_width; c++) {
            #pragma HLS UNROLL
            row_buf[0][c] = row1[c];
            row_buf[1][c] = row0[c];
        }
        for (unsigned b = 2; b < CONFIG_T::filt_height; b++) {
            #pragma HLS UNROLL
            for (unsigned c = 0; c < CONFIG_T::in_width; c++) {
                #pragma HLS UNROLL
                row_buf[b][c] = old_buf[b - 2][c];
            }
        }

        const bool have_window = i_pair >= PAIR_WINDOW_START;
        const bool emit_row0 = have_window && pair_phase == EMIT_ROW0_PHASE;
        const bool emit_row1 = have_window && pair_phase == EMIT_ROW1_PHASE;
        if (emit_row0 || emit_row1) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        WriteOutputWidth:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::out_width; i_iw++) {
                #pragma HLS UNROLL
                data_value_t kernel_data[CONFIG_T::filt_height];
                #pragma HLS ARRAY_PARTITION variable=kernel_data complete
                for (unsigned k = 0; k < CONFIG_T::filt_height - 2; k++) {
                    #pragma HLS UNROLL
                    kernel_data[k] = emit_row0
                        ? old_buf[CONFIG_T::filt_height - 2 - k][i_iw]
                        : old_buf[CONFIG_T::filt_height - 3 - k][i_iw];
                }
                kernel_data[CONFIG_T::filt_height - 2] = emit_row0 ? old_buf[0][i_iw] : row0[i_iw];
                kernel_data[CONFIG_T::filt_height - 1] = emit_row0 ? row0[i_iw] : row1[i_iw];

                res_value_t res_out[CONFIG_T::n_filt];
                #pragma HLS ARRAY_PARTITION variable=res_out complete
                CONFIG_T::mult_config::template kernel<
                    data_value_t, res_value_t, typename CONFIG_T::mult_config>::dense(
                        kernel_data, res_out, weights, biases);
                for (unsigned i_f = 0; i_f < CONFIG_T::n_filt; i_f++) {
                    #pragma HLS UNROLL
                    res_pack[i_iw * CONFIG_T::n_filt + i_f] = res_out[i_f];
                }
            }
            res.write(res_pack);
        }
        pair_phase = pair_phase == CONFIG_T::stride_height - 1 ? 0 : pair_phase + 1;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void maxpool2d_wide_nonoverlap_cl(
    hls::stream<data_T> &data, hls::stream<res_T> &res
) {
    static_assert(CONFIG_T::pool_height == 2 && CONFIG_T::stride_height == 2,
                  "Aria pooling requires non-overlapping 2x1 windows");
    static_assert(CONFIG_T::pool_width == 1 && CONFIG_T::stride_width == 1,
                  "Aria pooling requires width 1");
    static_assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 &&
                  CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0,
                  "Aria pooling does not support padding");
    static_assert(CONFIG_T::pool_op == nnet::Max, "Aria pooling requires Max");
    static_assert(data_T::size == CONFIG_T::in_width * CONFIG_T::n_filt,
                  "Aria pool input packing mismatch");
    static_assert(res_T::size == CONFIG_T::out_width * CONFIG_T::n_filt,
                  "Aria pool output packing mismatch");

    typename data_T::value_type previous[data_T::size];
    #pragma HLS ARRAY_PARTITION variable=previous complete
    bool second_row = false;
PoolRows:
    for (unsigned row = 0; row < CONFIG_T::in_height; row++) {
        #pragma HLS PIPELINE II=1
        data_T current = data.read();
        if (!second_row) {
            for (unsigned i = 0; i < data_T::size; i++) {
                #pragma HLS UNROLL
                previous[i] = current[i];
            }
        } else {
            res_T output;
            PRAGMA_DATA_PACK(output)
            for (unsigned i = 0; i < res_T::size; i++) {
                #pragma HLS UNROLL
                output[i] = previous[i] > current[i] ? previous[i] : current[i];
            }
            res.write(output);
        }
        second_row = !second_row;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_wide_stream(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out]
) {
    static_assert(CONFIG_T::n_out == 1, "Aria dense specialization requires one output");
    static_assert(res_T::size == 1, "Aria dense result must contain one value");
    static_assert(CONFIG_T::n_in % data_T::size == 0, "Aria dense input packing mismatch");
    static_assert(data_T::size % 7 == 0, "Aria dense expects seven filters per width lane");
    constexpr unsigned WIDTH_LANES = data_T::size / 7;
    constexpr unsigned WORDS = CONFIG_T::n_in / data_T::size;
    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    typename CONFIG_T::accum_t accumulator = (typename CONFIG_T::accum_t)biases[0];
    data_T input_word;
    unsigned word = 0;
    unsigned width = 0;
DenseValues:
    for (unsigned i = 0; i < WORDS * WIDTH_LANES; i++) {
        #pragma HLS PIPELINE II=1
        if (width == 0) {
            input_word = data.read();
        }
        for (unsigned filter = 0; filter < 7; filter++) {
            #pragma HLS UNROLL
            const unsigned data_index = width * 7 + filter;
            const unsigned weight_index = word * data_T::size + data_index;
            accumulator += (typename CONFIG_T::accum_t)CONFIG_T::template product<
                typename data_T::value_type, typename CONFIG_T::weight_t>::product(
                    input_word[data_index], weights[weight_index]);
        }
        if (width == WIDTH_LANES - 1) {
            width = 0;
            word++;
        } else {
            width++;
        }
    }
    res_T output;
    PRAGMA_DATA_PACK(output)
    output[0] = cast<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(accumulator);
    res.write(output);
}

} // namespace nnet

#endif
