#ifndef NNET_FIRST_CONV_STREAM_H_
#define NNET_FIRST_CONV_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_dense.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void first_conv_4lane_temporal_cl(
    hls::stream<data_T> &data, hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    static_assert(CONFIG_T::n_chan == 1, "first_conv_4lane_temporal_cl expects n_chan == 1");
    static_assert(CONFIG_T::filt_width == 1, "first_conv_4lane_temporal_cl expects filt_width == 1");
    static_assert(CONFIG_T::stride_width == 1, "first_conv_4lane_temporal_cl expects stride_width == 1");
    static_assert(CONFIG_T::in_width == data_T::size, "input pack width must match CONFIG_T::in_width");
    static_assert(CONFIG_T::n_filt == res_T::size, "output pack width must match CONFIG_T::n_filt");

    typedef typename data_T::value_type data_value_t;
    typedef typename res_T::value_type res_value_t;

    // Ring buffer: each iteration writes one slot; no shift, no loop-carried RAW dependency.
    data_value_t row_buf[CONFIG_T::filt_height][CONFIG_T::in_width];
    #pragma HLS ARRAY_PARTITION variable=row_buf complete dim=0

    // Use plain unsigned for array indices to avoid HLS ap_uint bit-extension warnings.
    // wptr: next slot to write (oldest slot after full window).
    // stride_cnt: replaces % stride_height, eliminates synthesized urem divider.
    unsigned wptr = 0;
    unsigned stride_cnt = 0;

ReadInputHeight:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=row_buf inter false

        data_T in_pack = data.read();

    InsertRow:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS UNROLL
            row_buf[wptr][i_iw] = in_pack[i_iw];
        }

        // oldest points to the slot just after wptr (the oldest row in the window).
        unsigned oldest = (wptr == CONFIG_T::filt_height - 1) ? 0u : wptr + 1;

        const bool have_full_window = i_ih >= CONFIG_T::filt_height - 1;
        const bool on_stride = have_full_window && (stride_cnt == 0);

        if (on_stride) {
        WriteOutputWidth:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
                #pragma HLS PIPELINE II=1

                data_value_t kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
                #pragma HLS ARRAY_PARTITION variable=kernel_data complete

                res_value_t res_out[CONFIG_T::n_filt];
                #pragma HLS ARRAY_PARTITION variable=res_out complete

                res_T res_pack;
                PRAGMA_DATA_PACK(res_pack)

            CopyKernel:
                for (unsigned k = 0; k < CONFIG_T::filt_height; k++) {
                    #pragma HLS UNROLL
                    // Circular read: oldest row is kernel row 0, newest is filt_height-1.
                    unsigned ridx = (oldest + k < CONFIG_T::filt_height)
                                    ? oldest + k
                                    : oldest + k - CONFIG_T::filt_height;
                    kernel_data[k] = row_buf[ridx][i_iw];
                }

                CONFIG_T::mult_config::template kernel<data_value_t, res_value_t, typename CONFIG_T::mult_config>::dense(
                    kernel_data, res_out, weights, biases);

            PackOutput:
                for (unsigned i_f = 0; i_f < CONFIG_T::n_filt; i_f++) {
                    #pragma HLS UNROLL
                    res_pack[i_f] = res_out[i_f];
                }

                res.write(res_pack);
            }
        }

        // Advance ring pointer to the oldest slot (next write overwrites oldest data).
        wptr = oldest;
        if (have_full_window) {
            stride_cnt = (stride_cnt == CONFIG_T::stride_height - 1) ? 0u : stride_cnt + 1;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void first_conv_4lane_temporal_wide_cl(
    hls::stream<data_T> &data, hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    static_assert(CONFIG_T::n_chan == 1, "first_conv_4lane_temporal_wide_cl expects n_chan == 1");
    static_assert(CONFIG_T::filt_width == 1, "first_conv_4lane_temporal_wide_cl expects filt_width == 1");
    static_assert(CONFIG_T::stride_width == 1, "first_conv_4lane_temporal_wide_cl expects stride_width == 1");
    static_assert(CONFIG_T::in_width == data_T::size, "input pack width must match CONFIG_T::in_width");
    static_assert(CONFIG_T::n_filt * CONFIG_T::out_width == res_T::size,
                  "wide output must pack all width positions and filters");

    typedef typename data_T::value_type data_value_t;
    typedef typename res_T::value_type res_value_t;

    data_value_t row_buf[CONFIG_T::filt_height][CONFIG_T::in_width];
    #pragma HLS ARRAY_PARTITION variable=row_buf complete dim=0

    unsigned wptr = 0;
    unsigned stride_cnt = 0;

ReadInputHeightWide:
    for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=row_buf inter false

        data_T in_pack = data.read();

    InsertRowWide:
        for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            #pragma HLS UNROLL
            row_buf[wptr][i_iw] = in_pack[i_iw];
        }

        unsigned oldest = (wptr == CONFIG_T::filt_height - 1) ? 0u : wptr + 1;

        const bool have_full_window = i_ih >= CONFIG_T::filt_height - 1;
        const bool on_stride = have_full_window && (stride_cnt == 0);

        if (on_stride) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)

        WriteOutputWidthWide:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::out_width; i_iw++) {
                #pragma HLS UNROLL

                data_value_t kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
                #pragma HLS ARRAY_PARTITION variable=kernel_data complete

                res_value_t res_out[CONFIG_T::n_filt];
                #pragma HLS ARRAY_PARTITION variable=res_out complete

            CopyKernelWide:
                for (unsigned k = 0; k < CONFIG_T::filt_height; k++) {
                    #pragma HLS UNROLL
                    unsigned ridx = (oldest + k < CONFIG_T::filt_height)
                                    ? oldest + k
                                    : oldest + k - CONFIG_T::filt_height;
                    kernel_data[k] = row_buf[ridx][i_iw];
                }

                CONFIG_T::mult_config::template kernel<data_value_t, res_value_t, typename CONFIG_T::mult_config>::dense(
                    kernel_data, res_out, weights, biases);

            PackWideOutput:
                for (unsigned i_f = 0; i_f < CONFIG_T::n_filt; i_f++) {
                    #pragma HLS UNROLL
                    res_pack[i_iw * CONFIG_T::n_filt + i_f] = res_out[i_f];
                }
            }

            res.write(res_pack);
        }

        wptr = oldest;
        if (have_full_window) {
            stride_cnt = (stride_cnt == CONFIG_T::stride_height - 1) ? 0u : stride_cnt + 1;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void first_conv_2row_4lane_temporal_wide_cl(
    hls::stream<data_T> &data, hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    static_assert(CONFIG_T::n_chan == 1, "first_conv_2row_4lane_temporal_wide_cl expects n_chan == 1");
    static_assert(CONFIG_T::filt_width == 1, "first_conv_2row_4lane_temporal_wide_cl expects filt_width == 1");
    static_assert(CONFIG_T::stride_width == 1, "first_conv_2row_4lane_temporal_wide_cl expects stride_width == 1");
    static_assert(CONFIG_T::in_width * 2 == data_T::size, "input pack must contain two full 4-lane rows");
    static_assert(CONFIG_T::n_filt * CONFIG_T::out_width == res_T::size,
                  "wide output must pack all width positions and filters");
    static_assert(CONFIG_T::in_height % 2 == 0, "2-row input expects an even input height");
    static_assert(CONFIG_T::stride_height >= 2,
                  "pair parallelism requires stride_height >= 2 to guarantee at most one output per pair");

    typedef typename data_T::value_type data_value_t;
    typedef typename res_T::value_type res_value_t;

    // Shift-register row buffer: buf[0]=newest row, buf[filt_height-1]=oldest row.
    // Pair parallelism: each iteration advances the buffer by 2 rows.  The new
    // state is computed entirely from the registered old_buf snapshot and the
    // current stream inputs — no intra-iteration RAW chain on row_buf → II=1.
    data_value_t row_buf[CONFIG_T::filt_height][CONFIG_T::in_width];
    #pragma HLS ARRAY_PARTITION variable=row_buf complete dim=0

    // pair_phase cycles 0..stride_height-1, one step per pair iteration.
    // For config4 (filt_height=5, stride_height=3):
    //   EMIT_ROW0_PHASE = 2: output row0 of the current pair  (row 2*i_pair)
    //   EMIT_ROW1_PHASE = 0: output row1 of the current pair  (row 2*i_pair+1)
    //   phase 1: no output
    //
    // General derivation: the first output row is filt_height-1 = 4 (even → row0).
    // It lands in pair (filt_height-1)/2 = 2, so pair_phase = 2 % stride_height = 2.
    // The next output (stride_height rows later) lands in pair
    // (filt_height-1+stride_height)/2 = 3, pair_phase = 3 % stride_height = 0.
    constexpr unsigned EMIT_ROW0_PHASE =
        ((CONFIG_T::filt_height - 1) / 2) % CONFIG_T::stride_height;
    constexpr unsigned EMIT_ROW1_PHASE =
        ((CONFIG_T::filt_height - 1 + CONFIG_T::stride_height) / 2) % CONFIG_T::stride_height;

    // Minimum pair index before the first complete filter window is loaded.
    constexpr unsigned PAIR_WINDOW_START = CONFIG_T::filt_height / 2;  // = 2 for filt_height=5

    unsigned pair_phase = 0;

    // Each iteration reads one stream word (row0 + row1), advances row_buf by 2
    // rows, and emits at most one output.  Trip count = in_height/2 = 128.
    //
    // Loop-carried dep: row_buf written at stage 0 (pure combinational unroll),
    // read at stage 0 of the next iteration → II=1 achievable.
ReadPairsWide:
    for (unsigned i_pair = 0; i_pair < CONFIG_T::in_height / 2; i_pair++) {
        #pragma HLS PIPELINE II=1

        data_T in_pack = data.read();

        data_value_t row0[CONFIG_T::in_width], row1[CONFIG_T::in_width];
        #pragma HLS ARRAY_PARTITION variable=row0 complete
        #pragma HLS ARRAY_PARTITION variable=row1 complete

        for (unsigned c = 0; c < CONFIG_T::in_width; c++) {
            #pragma HLS UNROLL
            row0[c] = in_pack[c];
            row1[c] = in_pack[CONFIG_T::in_width + c];
        }

        // Snapshot the current registered state.  old_buf is wired directly to
        // the row_buf registers — no extra pipeline stage, no intra-iteration
        // dependency between this read and the row_buf write below.
        data_value_t old_buf[CONFIG_T::filt_height][CONFIG_T::in_width];
        #pragma HLS ARRAY_PARTITION variable=old_buf complete dim=0

        for (unsigned b = 0; b < CONFIG_T::filt_height; b++) {
            #pragma HLS UNROLL
            for (unsigned c = 0; c < CONFIG_T::in_width; c++) {
                #pragma HLS UNROLL
                old_buf[b][c] = row_buf[b][c];
            }
        }

        // Advance row_buf by 2 rows unconditionally.  All sources are registered
        // values (old_buf) or current stream inputs; no sequential dependency
        // within the iteration.
        //   new_buf[0] = row1   (newest)
        //   new_buf[1] = row0
        //   new_buf[b] = old_buf[b-2]  for b = 2..filt_height-1  (oldest preserved)
    AdvanceBufPair:
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

        const bool have_window = (i_pair >= PAIR_WINDOW_START);
        const bool emit_row0   = have_window && (pair_phase == EMIT_ROW0_PHASE);
        const bool emit_row1   = have_window && (pair_phase == EMIT_ROW1_PHASE);

        if (emit_row0 || emit_row1) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)

        WriteOutputPairWide:
            for (unsigned i_iw = 0; i_iw < CONFIG_T::out_width; i_iw++) {
                #pragma HLS UNROLL

                // Build kernel[0..filt_height-1] (oldest→newest) from registered state.
                //
                // emit_row0 window = [old_buf[fh-2], ..., old_buf[0], row0]
                //   kernel[k] = old_buf[filt_height-2-k]  for k < filt_height-1
                //   kernel[filt_height-1] = row0
                //
                // emit_row1 window = [old_buf[fh-3], ..., old_buf[0], row0, row1]
                //   kernel[k] = old_buf[filt_height-3-k]  for k < filt_height-2
                //   kernel[filt_height-2] = row0
                //   kernel[filt_height-1] = row1
                data_value_t kernel_data[CONFIG_T::filt_height];
                #pragma HLS ARRAY_PARTITION variable=kernel_data complete

            BuildKernelPairWide:
                for (unsigned k = 0; k < CONFIG_T::filt_height - 2; k++) {
                    #pragma HLS UNROLL
                    kernel_data[k] = emit_row0
                        ? old_buf[CONFIG_T::filt_height - 2 - k][i_iw]
                        : old_buf[CONFIG_T::filt_height - 3 - k][i_iw];
                }
                kernel_data[CONFIG_T::filt_height - 2] =
                    emit_row0 ? old_buf[0][i_iw] : row0[i_iw];
                kernel_data[CONFIG_T::filt_height - 1] =
                    emit_row0 ? row0[i_iw] : row1[i_iw];

                res_value_t res_out[CONFIG_T::n_filt];
                #pragma HLS ARRAY_PARTITION variable=res_out complete

                CONFIG_T::mult_config::template kernel<data_value_t, res_value_t, typename CONFIG_T::mult_config>::dense(
                    kernel_data, res_out, weights, biases);

            PackPairOutput:
                for (unsigned i_f = 0; i_f < CONFIG_T::n_filt; i_f++) {
                    #pragma HLS UNROLL
                    res_pack[i_iw * CONFIG_T::n_filt + i_f] = res_out[i_f];
                }
            }

            res.write(res_pack);
        }

        pair_phase = (pair_phase == CONFIG_T::stride_height - 1) ? 0u : pair_phase + 1;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void unpack_4lane_temporal_cl(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    static_assert(CONFIG_T::n_filt == res_T::size, "narrow output must pack one width position of filters");
    static_assert(CONFIG_T::out_width * CONFIG_T::n_filt == data_T::size,
                  "wide input must pack all width positions and filters");

    data_T in_pack;
    PRAGMA_DATA_PACK(in_pack)
    unsigned i_iw = 0;

UnpackOutputFlat:
    for (unsigned i = 0; i < CONFIG_T::out_height * CONFIG_T::out_width; i++) {
        #pragma HLS PIPELINE II=1

        if (i_iw == 0) {
            in_pack = data.read();
        }

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)

    UnpackFilters:
        for (unsigned i_f = 0; i_f < CONFIG_T::n_filt; i_f++) {
            #pragma HLS UNROLL
            out_pack[i_f] = in_pack[i_iw * CONFIG_T::n_filt + i_f];
        }

        res.write(out_pack);

        i_iw = (i_iw == CONFIG_T::out_width - 1) ? 0u : i_iw + 1;
    }
}

} // namespace nnet

#endif
