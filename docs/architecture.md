# RAVEL architecture

RAVEL transforms a compatible trained model into a specialized FPGA inference
project while preserving the model semantics. Training remains outside RAVEL.

## Aria workflow

Aria 1.3.0 exposes one Python-first conversion path:

1. `ravel_hls.convert(model, config)` validates `Project`, `HLS`, optional
   `Optimization`, `Verification`, and `Vitis` settings.
2. hls4ml creates a clean baseline in a staging directory.
3. RAVEL builds semantic and streaming representations.
4. RAVEL resolves omitted optimization axes to the versioned P4/D2 default and
   applies the legality-checked Aria pass sequence.
5. RAVEL renders, verifies, and atomically publishes the project.
6. If `Vitis.Run` is true, the published project runs Vitis HLS and records its
   measured report. Conversion never launches the vendor tool by default.

The published project remains useful independently through `Project.open`,
`refresh`, `build`, `record`, and `link`. A failed vendor run does not roll back
or delete the generated sources; it leaves a diagnostic log and does not create
new qualification evidence.

## Transformation model

The Semantic IR records topology, parameters, quantization, and fixed-point
semantics. The Streaming IR records packing, rates, buffers, parallel
allocation, interfaces, and legal fusion. Aria applies these passes in order:

1. `PackTemporalInput2x` or `PackTemporalInput4x`
2. `FuseRepackReshapeIntoFirstConv`
3. `PropagateWideReLUStream`
4. `SpecializeNonOverlappingMaxPool`
5. `StreamFlattenIntoDense`
6. `BindShallowInternalFifos`
7. `ElideDataflowStartPropagation`

P2 carries two chronological rows in each 128-bit input word; P4 carries four
rows in each 256-bit word. Both use the same 28-value internal stream. P4 may
deassert input `TREADY` while draining a second convolution output row. Dense
x1 consumes one seven-filter group per step and Dense x2 consumes two while
retaining fixed-point accumulation order.

RAVEL renders owned files from the resolved plan through strict templates;
unaffected project files remain hls4ml-owned.

## Identity and integrity

Generation identity separates the semantic model, generation-affecting
configuration, and implementation. Model parameters and learned quantizer
state participate in semantic identity. Output location, verification choices,
and whether Vitis was invoked do not change the generation fingerprint.
The aggressive-policy identity and resolved P/D values do. Refresh reuses the
recorded resolved values and changes model state without changing architecture.

The manifest records a bounded source closure. Full inspection hashes only
generation-relevant files and prunes hidden directories and vendor `*_prj`
trees before traversal. Fast inspection intentionally reports
`source_integrity: not_checked`. Vendor evidence additionally binds the complete
manifest hash, generation fingerprint, source-closure hash, top, part, target
clock, tool version, and expected RTL port widths.

## Verification layers

- Structural validation checks the profile, configuration, templates, and
  generated-source contracts.
- Transformation equivalence checks bit-exact baseline and optimized C++ output
  for identical inputs.
- Model fidelity reports Keras/HGQ-to-HLS numerical agreement without a global
  promotion threshold.
- Performance qualification records Vitis HLS measurements without target
  pass/fail limits.

RTL simulation, IP export, implementation timing, and board validation remain
separate activities. Aria 1.3.0 does not promote HLS synthesis into proof of any
of those layers.
