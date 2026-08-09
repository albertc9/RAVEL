#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_types.h"
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_latency_wrapper(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                           typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                           typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_resource_wrapper(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                            typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

template <class data_T, typename CONFIG_T>
void data_prepare(hls::stream<data_T> &data_stream, typename data_T::value_type data[CONFIG_T::n_in]) {
    #pragma HLS INLINE

    if (CONFIG_T::n_in / data_T::size > 1) {
    DataPrepare:
        for (int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
            #pragma HLS PIPELINE
            data_T data_pack = data_stream.read();
        DataPackPipeline:
            for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
                #pragma HLS UNROLL
                data[i_in * data_T::size + i_pack] = data_pack[i_pack];
            }
        }
    } else {
        data_T data_pack = data_stream.read();
    DataPackSingle:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_pack] = data_pack[i_pack];
        }
    }
}

template <class res_T, typename CONFIG_T>
void res_write(typename res_T::value_type res[CONFIG_T::n_out], hls::stream<res_T> &res_stream) {
    #pragma HLS INLINE

    if (CONFIG_T::n_out / res_T::size > 1) {
    ResWrite:
        for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
            #pragma HLS PIPELINE
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        ResPackPipeline:
            for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = res[i_out * res_T::size + i_pack];
            }
            res_stream.write(res_pack);
        }
    } else {
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPackSingle:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_pack];
        }
        res_stream.write(res_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void dense_wide_stream(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
                       typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                       typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    #pragma HLS INLINE recursive

    static_assert(CONFIG_T::n_out == 1, "dense_wide_stream currently supports n_out == 1");
    static_assert(res_T::size == CONFIG_T::n_out, "result pack size must match n_out");
    static_assert(CONFIG_T::n_in % data_T::size == 0, "dense input size must be a whole number of wide words");
    static_assert(data_T::size % 7 == 0, "dense_wide_stream expects 7 filters per width lane");

    const unsigned n_words = CONFIG_T::n_in / data_T::size;
    const unsigned n_width_lanes = data_T::size / 7;

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc = (typename CONFIG_T::accum_t)biases[0];

    data_T in_pack;
    PRAGMA_DATA_PACK(in_pack)
    unsigned i_word = 0;
    unsigned i_width = 0;

DenseWideMain:
    for (unsigned i = 0; i < n_words * n_width_lanes; i++) {
        #pragma HLS PIPELINE II=1

        if (i_width == 0) {
            in_pack = data_stream.read();
        }

    DenseWideFilter:
        for (unsigned i_f = 0; i_f < 7; i_f++) {
            #pragma HLS UNROLL
            const unsigned data_idx = i_width * 7 + i_f;
            const unsigned weight_idx = i_word * data_T::size + data_idx;
            acc += (typename CONFIG_T::accum_t)CONFIG_T::template product<typename data_T::value_type,
                                                                          typename CONFIG_T::weight_t>::product(
                in_pack[data_idx], weights[weight_idx]);
        }

        if (i_width == n_width_lanes - 1) {
            i_width = 0;
            i_word++;
        } else {
            i_width++;
        }
    }

    res_T res_pack;
    PRAGMA_DATA_PACK(res_pack)
    res_pack[0] = cast<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(acc);
    res_stream.write(res_pack);
}

template <class data_T, class res_T, typename CONFIG_T>
void dense(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
           typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
           typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    #pragma HLS INLINE recursive

    typename data_T::value_type data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=res complete

    data_prepare<data_T, CONFIG_T>(data_stream, data);
    if (CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic) {
        dense_latency_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights,
                                                                                                  biases);
    }
    res_write<res_T, CONFIG_T>(res, res_stream);
}

} // namespace nnet

#endif
