#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<9,4,AP_RND,AP_SAT_SYM,0>, 4*1> input_layer_t;
typedef nnet::array<ap_fixed<9,4,AP_RND,AP_SAT_SYM,0>, 2*4> input_layer_x2_t;
typedef nnet::array<ap_fixed<9,4>, 1*1> reshape_t;
typedef ap_fixed<16,6> q_conv2d_accum_t;
typedef nnet::array<ap_fixed<16,6>, 7*1> q_conv2d_t;
typedef nnet::array<ap_fixed<16,6>, 7*4> q_conv2d_x4_t;
typedef ap_fixed<6,1> q_conv2d_weight_t;
typedef ap_fixed<5,1> q_conv2d_bias_t;
typedef nnet::array<ap_ufixed<15,5>, 7*1> q_conv2d_relu_t;
typedef nnet::array<ap_ufixed<15,5>, 7*4> q_conv2d_relu_x4_t;
typedef ap_fixed<18,8> q_conv2d_relu_table_t;
typedef ap_ufixed<15,5> max_pooling2d_accum_t;
typedef nnet::array<ap_fixed<9,4,AP_RND,AP_SAT_SYM,0>, 7*1> max_pooling2d_t;
typedef nnet::array<ap_fixed<9,4,AP_RND,AP_SAT_SYM,0>, 7*4> max_pooling2d_x4_t;
typedef ap_fixed<22,11> q_dense_accum_t;
typedef nnet::array<ap_fixed<22,11>, 1*1> result_t;
typedef ap_fixed<7,1> q_dense_weight_t;
typedef ap_fixed<4,0> q_dense_bias_t;
typedef ap_uint<1> layer9_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
