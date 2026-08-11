# Aria compatibility

## Model profile

Aria 1.3.0 accepts a single-input, single-output homogeneous HGQ model with this
semantic sequence:

```text
Input [256, 4]
  -> Reshape [256, 4, 1], channels last
  -> QConv2D(7 filters, 5x1 kernel, 3x1 stride, valid, ReLU)
  -> MaxPool2D(2x1 pool, 2x1 stride, valid)
  -> Flatten [1176]
  -> QDense(1, linear)
```

Weights, biases, legal learned K/I/F values, sparsity, and layer names may vary.
Geometry, connectivity, data format, input/output count, and the static
quantizer contract are compatibility requirements. This model profile is a
generation-legality contract, not a performance target.

## hls4ml and host profile

The optimized path requires the Vitis backend, `io_stream`, latency strategy,
and reuse factor 1. Project name, output path, FPGA part, clock period, model
parameters, verification inputs, and Vitis invocation remain user-selected.

`Optimization.TemporalPacking` accepts 2 or 4 and
`Optimization.DenseParallelism` accepts 1 or 2. Both are generation-time
choices. Missing axes resolve independently to P4 and D2. P2/D1 preserves the
Aria 1.1 input width and schedule semantics; P4 changes expected input TDATA
from 128 to 256 bits. Refresh preserves the recorded selection.

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

A `.ravelparams` package may update kernel, bias, and learned K/I/F state. Its
topology, canonical slot schema, shapes, dtypes, frontend contract, and static
quantizer type/rounding/overflow/axis/granularity must match the project-local
model template exactly. Static-contract changes require a complete model
refresh.

Packages contain no pickle or custom executable objects and reject traversal,
absolute paths, symlinks, duplicate entries, object arrays, invalid digests, and
oversized payloads. The archive is portable but unencrypted.
