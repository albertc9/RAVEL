# Aria 1.8.0 separable-block calibration

Development measurements for constant-matrix coverage version 3. All vendor
runs use Vitis HLS/Vivado 2023.2, KU5P `xcku5p-ffvb676-2-e`, 5 ns, external
P8/D4. Evidence layer: HLS synthesis, RTL CoSim and the RTL protocol test.
No routed implementation, board or multi-core result is claimed here.

## Predicted and measured intervals

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

## Coverage added in version 3

Kernels of one to seven rows, strides of one to three rows, blocks without a
pool, 11-bit inputs and first stages with a fill-cycle row schedule. Width 4,
at most 12 channels and filters, four position lanes and reuse factors 1 and 2
are unchanged. `calibration.json` records every measurement with its manifest,
source-closure and report hashes; `calibration_reports/` holds the top-level
HLS reports.

Every project passed C verification with intermediate boundaries, RTL CoSim
and the protocol test. Two protocol runs first reported no output stall because
the fixed 17-cycle backpressure pattern stayed in phase with the design; they
pass with the sliding phase introduced in the same development.
