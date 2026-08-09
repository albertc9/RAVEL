#include <iostream>

#include "cnn_core.h"
#include "parameters.h"


void cnn_core(
    hls::stream<input_layer_x2_t> &input_layer,
    hls::stream<result_t> &layer9_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_layer,layer9_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<q_conv2d_weight_t, 35>(w4, "w4.txt");
        nnet::load_weights_from_txt<q_conv2d_bias_t, 7>(b4, "b4.txt");
        nnet::load_weights_from_txt<q_dense_weight_t, 1176>(w9, "w9.txt");
        nnet::load_weights_from_txt<q_dense_bias_t, 1>(b9, "b9.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<q_conv2d_x4_t> layer4x4_out("layer4x4_out");
    #pragma HLS STREAM variable=layer4x4_out depth=4
    #pragma HLS BIND_STORAGE variable=layer4x4_out type=fifo impl=srl

    hls::stream<q_conv2d_relu_x4_t> layer5x4_out("layer5x4_out");
    #pragma HLS STREAM variable=layer5x4_out depth=4
    #pragma HLS BIND_STORAGE variable=layer5x4_out type=fifo impl=srl

    hls::stream<max_pooling2d_x4_t> layer6x4_out("layer6x4_out");
    #pragma HLS STREAM variable=layer6x4_out depth=4
    #pragma HLS BIND_STORAGE variable=layer6x4_out type=fifo impl=srl

    nnet::first_conv_2row_4lane_temporal_wide_cl<input_layer_x2_t, q_conv2d_x4_t, config4>(input_layer, layer4x4_out, w4, b4); // repack_reshape + q_conv2d

    nnet::relu<q_conv2d_x4_t, q_conv2d_relu_x4_t, relu_config5>(layer4x4_out, layer5x4_out); // q_conv2d_relu

    nnet::maxpool2d_wide_nonoverlap_cl<q_conv2d_relu_x4_t, max_pooling2d_x4_t, config6>(layer5x4_out, layer6x4_out); // max_pooling2d

    nnet::dense_wide_stream<max_pooling2d_x4_t, result_t, config9>(layer6x4_out, layer9_out, w9, b9); // q_dense

}
