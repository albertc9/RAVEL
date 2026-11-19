# Aria 1.8.0 separable-block qualification

Implementation: `f4f25d7` on `devel`. No release or publication was performed.
All vendor runs use Vitis HLS/Vivado 2023.2, KU5P `xcku5p-ffvb676-2-e`, 5 ns,
external P8/D4, on the qualification host. The trained model of the target
structure was not available; `final_target` is a synthetic model of the same
shape with random weights, so its numbers are not a result for trained weights.

## Final projects, ordinary automatic selection

HLS supplies II and latency. Resources and WNS are routed out-of-context
results for one core; a passing 5 ns route establishes timing at 200 MHz.
Capacities are 216,960 LUT, 433,920 FF and 1,824 DSP.

| Project | Model | Predicted | HLS II | Latency | LUT | FF | DSP | BRAM | Routed WNS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `final_target` | Synthetic target shape: 8x7 step 2, depthwise 7 step 2, 1x1 to 8 | 69 | 67 | 77 | 14,172 (6.53%) | 12,809 (2.95%) | 32 (1.75%) | 0 | +0.343 ns |
| `final_m09de` | Package model 09, dense-equivalent | 48 | 48 | 59 | 30,080 (13.86%) | 22,130 (5.10%) | 48 (2.63%) | 0 | +0.582 ns |
| `final_m13` | Package model 13 (half rate, current layers) | 29 | 31 | 38 | 51,346 (23.67%) | 33,486 (7.72%) | 48 (2.63%) | 0 | +0.265 ns |
| `final_heldout` | Held-out synthetic separable geometry | 53 | 52 | 60 | not routed | | | | |
| `final_m05` | Package model 05 (K7/S2 in both convolutions) | 70 | 73 | 80 | not routed | | | | |
| `final_m06` | Package model 06 (second convolution K3/S1) | 49 | 50 | 57 | not routed | | | | |
| `final_m12` | Package model 12 (half rate, K5/S1 then K5/S2) | 69 | 72 | 78 | not routed | | | | |
| `final_m01_t85` | Package model 01, `TargetII` 85 | 64 | 66 | 73 | not routed | | | | |

Every final project passed C verification on the built-in corpus with
intermediate boundaries, RTL CoSim and the RTL protocol test (input gaps, output
backpressure, reset between epochs, in-flight abort). Package models also pass
the additive 1,000 supplied events in C. `final_m13` uses throughput-first
ranking; no `TargetII` was given, so no lower-LUT plan was requested.
`final.json` records manifest, source-closure and file hashes, the selected
stages and the routed figures.

Before this work the target shape was rejected at P8, and with the native
hls4ml second group it measures II 1492 (`target_native` below).

## Other packaged models and the Dense-32 shape

Made after the separable structure worked, at the commits recorded in
`final.json`. Same tools, part, clock and evidence layers as above.

| Project | Model | Predicted | HLS II | Latency | LUT | FF | DSP | BRAM tiles | Routed WNS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `final_m08` | Package model 08 (three convolutions) | 48 | 49 | 63 | 41,914 (19.32%) | 24,134 (5.56%) | 48 (2.63%) | 0 | +0.427 ns |
| `final2_m07` | Package model 07 (hidden Dense 16) | 50 | 51 | 58 | 50,125 (23.10%) | 31,916 (7.36%) | 200 (10.96%) | 14.5 | +0.077 ns |
| `final2_m07_t85` | Package model 07, `TargetII` 85 | 64 | 65 | 73 | 34,669 (15.98%) | 22,645 (5.22%) | 100 (5.48%) | 8 | +0.300 ns |
| `final2_dense32` | Synthetic Dense-32 design shape: 12x7 step 2, depthwise 7 step 2, 1x1 to 16, Dense 32 | 69 | 69 | 81 | 27,340 (12.60%) | 25,446 (5.86%) | 525 (28.78%) | 16 | +0.413 ns |
| `final_m09_grouped` | Package model 09 in its grouped form | 48 | 48 | 59 | not routed | | | | |
| `final_m10_p4d1` | Package model 10, P4/D1, delegated downstream stages | uncalibrated | 844 | 872 | not routed | | | | |

`final2_dense32` has random weights. Its Dense stage uses 512 multipliers for
28,672 products in 56 steps. `final_dense32` is the same model before the Dense
accumulator loop was reduced to one addition: HLS II 67, routed WNS -0.253 ns
with 468 failing endpoints, all in the Dense accumulators. It is kept as the
record of that timing miss. `final_m09_grouped` reads the grouped file directly;
the frontend rewrites it and the result equals `final_m09de`. Package model 10
has a kernel across the four antennas; no RAVEL stage lowers it, and Dense
parallelism 4 is not legal after the width collapses to one.

The Dense matrix stage measures 9 cycles more than its steps with 48 input
lanes and 12 with 64; the estimate uses 12.

## Calibration measurements

Development measurements for constant-matrix coverage version 3, made at
`b1bb8cb`. Evidence layer: HLS synthesis, RTL CoSim and the RTL protocol test.

### Predicted and measured intervals

Predicted values are the analytical frame cycles of the selected plan. The
held-out prediction was recorded before its run.

| Project | Model | Predicted | HLS II | Latency |
|---|---|---:|---:|---:|
| `dev_target_engines` | Synthetic target shape (8x7 step 2, depthwise 7 step 2, 1x1) | 69 | 67 | 77 |
| `dev_m09de_engines` | Package model 09, dense-equivalent | 48 | 48 | 59 |
| `dev_m09de_t85` | Package model 09, dense-equivalent, `TargetII` 85 | 62 | 62 | 79 |
| `dev_heldout_engines` | Held-out synthetic (192 rows, K5/S2, depthwise K3/S2, 1x1 to 10) | 53 | 52 | 60 |
| `dev_m05_engines` | Package model 05 (K7/S2 in both convolutions) | 70 | 73 | 80 |
| `dev_m06_engines` | Package model 06 (second convolution K3/S1) | 49 | 50 | 57 |
| `m13_p8d4` | Package model 13, ordinary selection | 29 | 31 | 38 |
| `m02_p8d4` | Package model 02, ordinary selection | 49 | 51 | 58 |
| `m03_p8d4` | Package model 03, ordinary selection | 60 | 51 | 58 |
| `m04_p8d4` | Package model 04, ordinary selection | 80 | 74 | 80 |

`dev_*` projects were made with coverage opened to measure candidates before
coverage was declared. They are development comparisons, not acceptance
evidence. The other rows use ordinary selection under version 2 coverage; 03
and 04 exercise the fill-cycle first-stage schedule with the base PHARA graph.

With a native hls4ml second group, the synthetic target shape measures II 1492
(`target_native`) and package model 09 measures II 844 (`m09de_native`). Their
native-stage estimates are uncalibrated lower bounds.

### Coverage added in version 3

Kernels of one to seven rows, strides of one to three rows, blocks without a
pool, 11-bit inputs and first stages with a fill-cycle row schedule. Width 4,
at most 12 channels, four position lanes and reuse factors 1 and 2 are
unchanged; filters reach 16 with the Dense-32 shape as evidence. `calibration.json` records every measurement with its manifest,
source-closure and report hashes; `calibration_reports/` holds the top-level
HLS reports.

Every project passed C verification with intermediate boundaries, RTL CoSim
and the protocol test. Two protocol runs first reported no output stall because
the fixed 17-cycle backpressure pattern stayed in phase with the design; they
pass with the sliding phase introduced in the same development.
