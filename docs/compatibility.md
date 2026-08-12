# Aria compatibility

## Model profile

Aria 1.5.0 recognizes a single-input, single-output homogeneous HGQ2 family with
this semantic sequence. Dimensions are symbols extracted from the converted
`ModelGraph`, not constants copied from one training archive:

```text
Input [H, W]
  -> Reshape [H, W, 1], channels last
  -> QConv2D(F filters, Khx1 kernel, Shx1 stride, valid, ReLU)
  -> MaxPool2D(2x1 pool, 2x1 stride, valid)
  -> Flatten [N]
  -> QDense(1, linear)
```

Weights, biases, legal learned fixed-point types, sparsity, and layer names may
vary. Geometry is extracted rather than compared with a single archive, while
each selected Aria strategy still enforces its own kernel, stride, packing, and
pooling legality. Connectivity, data format, input/output count, and static
quantizer semantics remain compatibility requirements. Recognition is not a
performance target.

Family recognition and strategy applicability are separate. P2 currently
requires `H` divisible by 2, `Kh >= 3`, and `Sh >= 2`; P4 requires `H`
divisible by 4 and the qualified `Kh=5`, `Sh=3` schedule. Both require one
input channel, width-one convolution, valid padding, the shown non-overlapping
MaxPool, one Dense output, and a Dense parallelism that divides the streamed
convolution width. An unsupported strategy returns structured findings before
rendering. The regression suite includes a P2 case with `[128,4]`, five filters,
a 3x1/stride-2 convolution, and `N=620`, in addition to the 12 retrained
canonical-geometry models.

## hls4ml and host profile

The optimized path requires the Vitis backend, `io_stream`, latency strategy,
and reuse factor 1. Project name, output path, FPGA part, clock period, model
parameters, verification inputs, and Vitis invocation remain user-selected.

`Optimization.TemporalPacking` accepts 2 or 4 and
`Optimization.DenseParallelism` accepts 1 or 2. Both are generation-time
choices. Missing axes resolve independently to P4 and D2. P2/D1 preserves the
Aria 1.1 input width and schedule semantics; P4 changes expected input TDATA
from 128 to 256 bits. Refresh preserves the recorded selection.

Aria 1.5 derives a sequential packed Dense weight ROM from the converted
hls4ml graph. Word width, depth, MAC lanes, and tail handling are internal plan
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

Successful synthesis is imported automatically. A report is accepted only when
its tool version, top, part, target clock, and expected stream port widths match
the immutable project identity. II, latency, estimated clock, and resources are
measurements: RAVEL does not require a particular II, does not require estimated
clock to beat the target, and does not define matrix-specific release gates.

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
