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

hls4ml 1.2.0's Vitis backend currently invokes `vitis-run`, while a standalone
Vitis HLS 2023.2 installation exposes the deprecated-but-supported
`vitis_hls` launcher instead. The generated `build_prj.tcl` remains compatible
with 2023.2 and can be run directly with that launcher after setting explicit
build options. RAVEL does not create a fake `vitis-run` shim.

## Dependency policy

RAVEL uses a project-specific virtual environment and never changes packages
at runtime. Aria 1.0 metadata declares exact compatibility-sensitive pins. The
CNN-for-Arianna reference uses exact compatibility-sensitive pins from
`constraints/aria-reference.txt`.

HGQ and HGQ2 distributions must not coexist in the reference environment:
both provide the `hgq` import namespace and installation order can silently
replace runtime files while leaving conflicting distribution metadata behind.
The qualified stack uses HGQ2 alone because the canonical model is a
Keras 3 artifact and loads successfully through HGQ2's `hgq` namespace.

On 2026-08-09, the clean Linux stack was qualified with CPython 3.11.15:

- all 45 installed distributions passed dependency consistency checking;
- the canonical model loaded as input `[256, 4]` and output `[1]`;
- hls4ml 1.2.0 recognized the expected HGQ layer sequence;
- the public RAVEL conversion completed with required bit-exact transformation
  equivalence;
- the published project reopened through `RavelProject.link_hls4ml`, compiled,
  and predicted 1000 supplied samples with maximum absolute difference `0`
  from the clean hls4ml baseline;
- the Keras/HGQ-to-HLS score fidelity on that run was `1.0000`.

This stage qualifies dependencies, generation, source-level C++ compilation,
and transformation correctness; by itself it carries no vendor-performance
claim. The preserved legacy Vitis report is historical comparison evidence and
cannot qualify a new RAVEL manifest.

The same date, a current RAVEL-generated project was synthesized independently
with Vitis HLS 2023.2 for `xcku5p-ffvb676-2-e` at a 5 ns target. The report
recorded II 178, latency 183 cycles, a 3.647 ns estimated clock, 4 DSP, 3483 FF,
28922 LUT, no BRAM/URAM, and 128/32-bit input/output TDATA. These measurements
qualify only the exact manifest linked by `ravel_qualification.json`; they are
not a universal Aria performance guarantee.

A one-transaction zero-input XSim RTL co-simulation also completed with
Verilog status `Pass`, and the generated validation step reported identical C
and RTL result files. This exercises one complete stream transaction without
backpressure; it is not multi-transaction, randomized-stall, IP-export,
implementation, or board validation.
