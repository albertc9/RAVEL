#include <iostream>

#include "cnn_core.h"
#include "parameters.h"


void cnn_core(
    hls::stream<input_layer_t> &input_layer,
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

    hls::stream<reshape_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1024

    hls::stream<q_conv2d_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=336

    hls::stream<q_conv2d_relu_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=336

    hls::stream<max_pooling2d_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=168

    auto& layer7_out = layer6_out;
    nnet::repack_stream<input_layer_t, reshape_t, 1024>(input_layer, layer10_out); // repack_reshape

    nnet::conv_2d_cl<reshape_t, q_conv2d_t, config4>(layer10_out, layer4_out, w4, b4); // q_conv2d

    nnet::relu<q_conv2d_t, q_conv2d_relu_t, relu_config5>(layer4_out, layer5_out); // q_conv2d_relu

    nnet::pooling2d_cl<q_conv2d_relu_t, max_pooling2d_t, config6>(layer5_out, layer6_out); // max_pooling2d

    nnet::dense<max_pooling2d_t, result_t, config9>(layer7_out, layer9_out, w9, b9); // q_dense

}

