# Aria compatibility

## Model profile

Aria 1.8.0 recognizes a single-input, single-output homogeneous HGQ2
family: a chain of temporal blocks, a Flatten, and a Dense head. Dimensions are
symbols extracted from the converted `ModelGraph`, not constants copied from
one training archive:

```text
Input [H, W]
  -> Reshape [H, W, 1], channels last
  -> one or more temporal blocks:
       QConv2D(F filters, KhxKw kernel, ShxSw stride, valid)
       [-> ReLU]
       [-> MaxPool2D(valid)]
  -> Flatten [N]
  -> QDense(M, linear)
  [-> ReLU -> QDense(1, linear)]
```

A temporal block is one convolution with an optional ReLU and an optional
MaxPool, in that order. A MaxPool requires the ReLU before it, and the last
block of the chain ends in a MaxPool. The Dense head is one Dense that consumes
the Flatten, optionally followed by ReLU and Dense pairs; the final Dense has
one output. The single-block form `Conv, ReLU, MaxPool, Flatten, Dense(1)` is
unchanged.

Recognition and qualification are separate. Aria 1.8.0 qualifies chains that

- start with a block that ends in ReLU and MaxPool;
- contain at most two blocks with a MaxPool;
- contain at most one block without a MaxPool, with or without a ReLU;
- have at most one hidden Dense layer.

A depthwise-separable layer falls under these rules: an activation-free
convolution without a pool, followed by a 1x1 convolution with ReLU and
MaxPool. Other recognized chains are reported as unsupported with a local
finding (`family.support.block_count`, `family.support.block_form`, or
`family.support.head_form`).

Weights, biases, legal learned fixed-point types, sparsity, and layer names may
vary. Geometry is extracted rather than compared with a single archive, while
each selected Aria strategy still enforces its own kernel, stride, packing, and
pooling legality. Connectivity, data format, input/output count, and static
quantizer semantics remain compatibility requirements. Recognition is not a
performance target.

Family recognition and strategy applicability are separate. P2 currently
requires `H` divisible by 2, `Kh >= 3`, and `Sh >= 2`; P4 requires `H`
divisible by 4 and the qualified `Kh=5`, `Sh=3` schedule. P8 requires `H`
divisible by 8 and derives its row schedule from the extracted kernel, stride,
and pool. The `Kh=5`, `Sh=3` geometry keeps its original P8 schedule and
sources. Other geometries use a fill-cycle schedule: it still emits at most one
pooled row per cycle and reads without producing when the next pooled row needs
rows that have not arrived.
P2 analysis derives a next-window-end schedule and accepts it only when the
calculated output count agrees with the extracted convolution geometry and no
two outputs require the same two-row input word.
The first-block specializations require one input channel, width-one
convolution, valid padding, and a 2x1 MaxPool with stride 2x1. Later blocks
accept channel mixing. Inside calibrated coverage they use RAVEL's
captured-window or shared-arithmetic stage, with or without ReLU and a
non-overlapping MaxPool; otherwise they use the pinned native hls4ml
Latency/ReuseFactor 1 implementation. A convolution whose kernel spans the stream width always uses
the native implementation. Dense parallelism must divide the stream width that
reaches the Dense head, and the final Dense must have one output. A hidden
Dense uses RAVEL's sequential matrix stage; the ReLU and Dense after it use the
native implementation. An unsupported strategy returns structured findings before
rendering. The regression suite includes a P2 case with `[128,4]`, five filters,
a 3x1/stride-2 convolution, and `N=620`, in addition to the 12 retrained
canonical-geometry models.
Position-sensitive P2 regression coverage includes `Kh=7`, `Sh=2`, leading and
trailing row impulses, and schedule-property checks across `Kh=3..8` and
`Sh=2..5`. Aria 1.8.0 adds P8 schedule-property checks across `Kh=1..11` and
`Sh=1..6`, and bit-exact chains with separable, unpooled, grouped, and
hidden-Dense forms.

## Source-model normalization

hls4ml 1.2.0 does not convert two layer forms of supported models correctly.
RAVEL expresses them in an equivalent form before conversion. The supplied
model file is not edited.

- A plain Keras `ReLU` layer, without slope, threshold, or maximum, becomes
  `Activation("relu")`. hls4ml converts the standalone layer as a thresholded
  ReLU whose C model passes negative values.
- An HGQ2 `QConv2D` with `groups > 1` becomes an ordinary convolution whose
  kernel is zero outside each filter's group. Quantizer state is copied
  unchanged. A grouped convolution that cannot be mapped this way is reported
  as `frontend.operation.grouped_convolution`.

A model that hls4ml refuses to convert is a compatibility result with the
finding `frontend.conversion.hls4ml`, not an internal error. This includes an
unquantized layer directly after the input, such as decimation inside the
model.

## hls4ml and host profile

The optimized path requires the Vitis backend, `io_stream`, latency strategy,
and reuse factor 1. Project name, output path, FPGA part, clock period, model
parameters, verification inputs, and Vitis invocation remain user-selected.

`Optimization.TemporalPacking` accepts 2, 4, or 8 and
`Optimization.DenseParallelism` accepts 1, 2, or 4. The supported pairs are
P2/D1, P2/D2, P4/D1, P4/D2, and P8/D4. Omission resolves to P8/D4. For the
canonical model, P2/P4/P8 use 128-/256-/512-bit input `TDATA`. Refresh preserves
the recorded selection.

`Optimization.TargetII` is an optional positive interval in clock cycles.
Without it the search ranks plans by predicted interval alone. For a model with
a hidden Dense this can select twice the multipliers for a gain of a few
cycles, so give a target when resources matter. With a target, the plan with
the fewest predicted LUT inside the target is selected.

Aria derives a sequential packed Dense weight ROM from the converted hls4ml
graph. A Dense with several outputs uses one ROM per output lane and computes a
searched number of outputs per step. Two outputs of a step share one multiplier
per input value when the packed operand fits the multiplier ports of the
device. Word width, depth, MAC lanes, outputs per
step, multiplier sharing, and tail handling are internal plan
properties; they are not additional public configuration fields. Refresh may
change parameter values but rejects changes to the recorded structural plan.

Linux supports the complete qualified workflow. macOS supports model parsing,
generation, post-processing, package handling, and inspection; automatic C++
verification may be unavailable when the HLS simulation headers cannot be
compiled. Windows is not supported.

The compatibility-sensitive Python stack is pinned in
`constraints/aria-reference.txt`. Use HGQ2 alone; the retired `HGQ` distribution
conflicts on the same Python namespace.

## Vitis HLS 2023.2

`Project.build()` invokes the standalone `vitis_hls` launcher directly with the
generated `build_prj.tcl`; it does not depend on hls4ml's newer `vitis-run`
adapter. RAVEL removes hls4ml's unsupported
`config_array_partition -maximum_size` command before publication. The default
stage profile resets the HLS project and runs synthesis only.

The P2/P4 first-convolution plan budgets the complete set of products needed by
every unrolled output-width window. PHARA instead records a fixed graph envelope
and a coefficient-dependent hybrid realization. The realization is accepted
only after modular symbolic equivalence is proven. Refresh preserves the
envelope and regenerates the graph; it does not reuse stale coefficient logic.

Successful synthesis is imported automatically. A report is accepted only when
its tool version, top, part, target clock, and expected stream port widths match
the immutable project identity. II, latency, estimated clock, and resources are
measurements: RAVEL does not require a particular II, does not require estimated
clock to beat the target, and does not define matrix-specific release gates.
If `Vitis.Stages.CoSim` is true, recording additionally requires a passing
top-level Verilog CoSim report and binds its hash into the qualification record.
For PHARA designs, recording also requires the unique fused-region, Dense
wrapper, and Dense pipeline reports. Their intervals and latencies are stored
as stage evidence.

This support does not strengthen the RTL proof boundary. CoSim, validation,
export, Vivado synthesis, implementation, and board tests run only when selected
by the user and retain their own evidence semantics.

## Parameter-package compatibility

A schema-v2 `.ravelparams` package carries ModelGraph kernel and bias payloads
under canonical operation/role IDs. Its model-structure fingerprint, shapes,
numeric descriptors, family, and static frontend provenance must match the
project architecture contract. A learned precision change requires ordinary
conversion.

Packages contain no pickle or custom executable objects and reject traversal,
absolute paths, symlinks, duplicate entries, object arrays, invalid digests, and
oversized payloads. The archive is portable but unencrypted.


## Composed plans and protocol verification

Planning uses canonical connectivity and dimensions. Layer names, archive names,
and target-specific extents do not select strategies. Flatten is a C-order
layout view with a reversible temporal/feature/channel index mapping. Padding,
branches, merges, extra outputs, opaque arithmetic, heterogeneous quantizers,
and unsupported schedules produce structured local findings before publication.
A complete production trace is bounded to 4096 output-word events per temporal
stage; exceeding the bound rejects that capability without truncation.

Internal bridges preserve scalar integer codes and channel order. They mask
only the final padding lanes. The composed linear blocking chain uses positive
FIFO capacity and assumes eventual input and output readiness. Native-stage
cycle estimates are uncalibrated lower bounds; HLS and routed measurements are
separate evidence. There is no universal II or clock-closure threshold.

Composed Vitis builds use `config_rtl -reset all` to clear the native streaming
coordinates and discard an interrupted inference. Existing single-block
renderer and reset settings remain unchanged. The optional executable protocol
checker drives `ap_start` until `ap_ready` accepts each transaction independently
of AXI input buffering; it checks exact output words, stable stalled outputs,
input gaps, mid-inference reset, reset between epochs, and consecutive frames.
It prefers Vivado 2023.2 XSim, then Verilator with timing support, then Icarus.
Verilator 5.040 encountered active-region convergence failures on the vendor's
full-reset RTL, so qualification uses XSim.

The mandatory `numeric-contract-v2` corpus is always present when C verification
runs. Supplied inputs are additive and are quantized once using the input
contract. Separate sample counts and hashes are recorded. Synthesized-RTL
reference words come from the clean C baseline and use the built-in corpus by
default. C stage observations are temporary instrumentation and are not emitted
in the published RTL interface.
