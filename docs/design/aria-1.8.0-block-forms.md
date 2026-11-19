# Aria 1.8.0: block forms, derived P8 schedule and hidden Dense

Aria 1.8.0 widens the temporal family. A block may omit its ReLU or its pool,
the Dense head may have one hidden layer, and P8 PHARA is no longer limited to
the K5/S3 first layer. Existing strategy selection, the generated sources of
earlier configurations and the recorded architectures of earlier projects are
preserved. External temporal packing and Dense parallelism remain caller choices.

## Temporal blocks

A temporal block is one convolution with an optional ReLU and an optional
non-overlapping MaxPool, in that order. A pool keeps its ReLU, and the chain
ends in a pooled block. Geometry is extracted; kernel and stride limits belong
to strategies, not to the grammar.

Qualified chains start with a pooled block, hold at most two pooled blocks and
at most one block without a pool. This includes

- a depthwise-separable layer: an activation-free convolution followed by a
  1x1 convolution with ReLU and MaxPool;
- three convolutions with pooling after the first and the third.

Other recognized chains return a `family.support.*` finding before rendering.

## Depthwise convolution

A depthwise or grouped convolution is an ordinary convolution whose kernel is
zero outside each filter's group. hls4ml 1.2.0 does not convert `groups > 1`,
so the frontend rewrites a grouped HGQ2 convolution into this form and copies
its quantizer state unchanged. A model already exported in this form is
accepted as it is. No grouped operator is added: the shared-arithmetic
optimizer builds its graph from the coefficient codes and creates no node for a
zero, so the stage costs only the nonzero products.

## P8 row schedule

P8 applicability follows from the row schedule of the extracted kernel height
`K`, stride `S` and pool height `P`. One pooled row spans `K + (P - 1) * S`
input rows and advances the input by `P * S` rows. The schedule emits at most
one pooled row per cycle. When the rows read so far do not cover the next
pooled row, the cycle reads one input word and produces nothing (a fill cycle).
Buffer depth, total cycles and fill cycles are computed from the geometry and
recorded in the plan.

The K5/S3 geometry keeps its previous schedule, lowering and source bytes.
Other geometries use a separate versioned lowering. P4 still requires K5/S3 and
P2 keeps its own schedule proof. Two pooled rows per cycle are not part of this
version.

The rate consequence is structural. A K7/S2 first layer on 256 rows produces 62
pooled rows per inference, against 42 for K5/S3, so its first stage needs at
least 62 cycles.

## Hidden Dense

The Dense head is a Dense that consumes the Flatten, optionally followed by
ReLU and Dense pairs; the final Dense has one output. One hidden layer is
qualified.

The hidden Dense is lowered by `aria-dense-matrix`. Each step reads one input
group and updates some of the outputs. Weights come from one ROM per output
lane, and every output accumulates its products in input order, as the native
Dense does. The input group is the requested Dense parallelism, or the largest
smaller one whose ROM word fits 1024 bits. The number of outputs per step is
enumerated by the search together with the convolution stages, so `TargetII`
can trade multipliers for steps. The following ReLU and the final Dense use the
native hls4ml implementation (`hls4ml-head-tail`); they carry one word per
inference.

hls4ml 1.2.0 converts a standalone Keras `ReLU` layer as a thresholded ReLU
whose C model passes negative values. The frontend expresses a plain `ReLU`
layer as `Activation("relu")`, which computes the same function and converts
correctly.

## Search and calibration boundaries

Complete-chain enumeration is bounded by 64 candidates for two blocks and by
4,096 for three blocks or for two blocks with a scheduled Dense head, a factor
of 64 for each further searched stage. The qualified chains are searched
completely. Hitting the bound still produces an incomplete-search
report, not a global-optimum claim.

Constant-matrix coverage version 4 targets KU5P at 5 ns with four position
lanes on streams of width four and up to 256 rows. It covers kernels of one to
seven rows, strides of one to three rows, blocks with or without a two-row
pool, inputs up to 12 bits, accumulators up to 24 bits, up to 12 input channels
and 16 filters, and arithmetic graphs with a fanout up to 17. Each limit is the
widest measured value. Reuse factors 1 and 2 are calibrated. Legal variants outside
coverage stay visible without automatic promotion. A convolution whose kernel
spans the stream width has no specialized stage and uses the delegated hls4ml
stage. The interval estimate of the Dense matrix stage is its step count plus a
measured overhead of 13 cycles. Resource estimates remain analytical
predictions, not placement or timing certificates.

## Records and refresh

Manifest schema 9 records the operations of each block, the row-credit
scheduler version with its cycle counts, the Dense matrix schedule and the
head tail. Earlier manifest versions remain readable. A project of an earlier
generation is refreshed under its recorded generation and keeps its stage and
arithmetic schedules.

## Outside this version

- A first convolution with more than one input channel.
- More than two pooled blocks, more than one block without a pool, or more than
  one hidden Dense layer.
- A specialized stage for a kernel across the stream width. Such a model
  converts at P4/D1 or P2/D1 through the delegated stage; Dense parallelism
  must divide the width that reaches the Dense head.
- An unquantized layer directly after the input, such as decimation inside the
  model. hls4ml refuses it and RAVEL reports `frontend.conversion.hls4ml`.

## Qualification

All results are single-core, KU5P at 5 ns, P8/D4, ordinary automatic selection.
II and latency are Vitis HLS 2023.2 synthesis results; resources and WNS are
routed out-of-context Vivado 2023.2 results. Every project passed C
verification against the clean hls4ml baseline, RTL CoSim and the protocol test.

| Model | II | Latency | LUT | FF | DSP | BRAM tiles | WNS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Depthwise-separable second layer (K5/S3), trained | 48 | 59 | 30,080 (13.86%) | 22,130 (5.10%) | 48 (2.63%) | 0 | +0.582 ns |
| Three convolutions (8/8/12 filters), trained | 49 | 63 | 41,914 (19.32%) | 24,134 (5.56%) | 48 (2.63%) | 0 | +0.427 ns |
| Hidden Dense of 16 units, trained | 51 | 58 | 50,125 (23.10%) | 31,916 (7.36%) | 200 (10.96%) | 14.5 | +0.077 ns |
| Hidden Dense of 16 units, trained, `TargetII=85` | 65 | 73 | 34,669 (15.98%) | 22,645 (5.22%) | 100 (5.48%) | 8 | +0.300 ns |
| 8x7 step 2, depthwise 7 step 2, 1x1 to 8; random weights | 67 | 77 | 14,172 (6.53%) | 12,809 (2.95%) | 32 (1.75%) | 0 | +0.343 ns |
| 12x7 step 2, depthwise 7 step 2, 1x1 to 16, Dense 32; random weights | 69 | 81 | 27,340 (12.60%) | 25,446 (5.86%) | 525 (28.78%) | 16 | +0.413 ns |

The last two models have the layer shapes of designs under study and random
weights; their resources are not results for trained weights. With the
delegated hls4ml second stage the first of them measures II 1492. The grouped
form of the depthwise model gives the same II and latency as its
block-diagonal export. These are source-bound reference results, not
predictions for arbitrary models. System integration, replicated cores, board
validation and model accuracy are outside this work. See the
[qualification evidence](../../references/qualification/aria_1_8_0_separable/README.md).
