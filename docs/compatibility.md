# Aria compatibility

## Model profile

Aria 1.0 accepts a single-input, single-output homogeneous HGQ model with this
semantic sequence:

```text
Input [256, 4]
  -> Reshape [256, 4, 1], channels last
  -> QConv2D(7 filters, 5x1 kernel, 3x1 stride, valid, ReLU)
  -> MaxPool2D(2x1 pool, 2x1 stride, valid)
  -> Flatten [1176]
  -> QDense(1, linear)
```

Weights, biases, sparsity, legal homogeneous precision parameters, and layer
names may vary. Geometry, connectivity, data format, input/output count, and
homogeneous quantization are compatibility requirements. A mathematically
similar but differently serialized topology is unsupported until separately
qualified.

## hls4ml project profile

The canonical optimized path requires:

- Vitis backend
- `io_stream`
- `Latency` strategy
- reuse factor 1

Output directory, project name, FPGA part, clock period, compatible precision,
model parameters, and verification settings remain selectable through their
owning APIs. Expressible hls4ml values are not automatically supported by the
Aria profile.

## Host support

Linux supports generation and non-vendor C++ verification. macOS supports
model parsing, project generation, post-processing, and inspection; automatic
behavioral verification may be unavailable when a supported compiler cannot
compile the required HLS simulation headers. Windows is not initially
supported.

Vendor-backed work follows the selected Vitis installation's platform rules
and is outside ordinary RAVEL conversion.

## Dependency policy

RAVEL uses a project-specific virtual environment and never changes packages
at runtime. Published metadata will declare finite tested ranges. The
CNN-for-Arianna reference uses exact compatibility-sensitive pins from
`constraints/aria-reference.txt`.

HGQ and HGQ2 distributions must not coexist in the reference environment:
both provide the `hgq` import namespace and installation order can silently
replace runtime files while leaving conflicting distribution metadata behind.
The current candidate stack uses HGQ2 alone because the canonical model is a
Keras 3 artifact and loads successfully through HGQ2's `hgq` namespace.

On 2026-08-09, the candidate Linux stack was checked with CPython 3.11.15:

- all 45 installed distributions passed dependency consistency checking;
- the canonical model loaded as input `[256, 4]` and output `[1]`;
- hls4ml 1.2.0 recognized the expected HGQ layer sequence;
- baseline project generation and C-simulation compilation passed;
- the legacy 1000-sample run reported HLS accuracy `0.9840`, HLS/Keras
  fidelity `1.0000`, and maximum absolute score difference `0`.

This is baseline dependency evidence only. The stack is not called qualified
until RAVEL's own transformation-equivalence and project-conformance suites
pass. It carries no Vitis synthesis, timing, resource, RTL-interface, or
initiation-interval claim.
