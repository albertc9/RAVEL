# RAVEL architecture

RAVEL transforms a compatible trained model into a specialized FPGA inference
project while preserving the model semantics. Training remains outside RAVEL.

## Aria workflow

Aria 1.5.0 exposes three Python operations over one configuration:

1. `analyze(model, config)` converts to a clean hls4ml `ModelGraph`, extracts
   immutable model facts, matches a versioned family, and resolves a design
   without publishing files.
2. `convert(model, output_dir, config)` reuses that same analysis, creates a
   clean baseline in a staging directory, renders from the resolved design and
   ModelGraph parameter payload, verifies it, and atomically publishes it.
3. `refresh(project, model_or_parameters)` requires the recorded architecture
   contract and atomically regenerates the project.
4. RAVEL resolves omitted optimization axes to the versioned P4/D2 default and
   executes the legality-checked Aria pass sequence.
5. If `Vitis.Run` is true, the published project runs Vitis HLS and records its
   measured report. Conversion never launches the vendor tool by default.

The published project remains useful independently through `Project.open`,
`refresh`, `build`, `record`, and `link`. A failed vendor run does not roll back
or delete the generated sources; it leaves a diagnostic log and does not create
new qualification evidence.

## Transformation model

The hls4ml `ModelGraph` is the authoritative compiler IR. RAVEL projects it into
immutable model facts and executes resolved-design transformations for packing,
rates, buffers, parallel allocation, interfaces, and legal fusion. Aria records
these actual transformations in order:

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

Before rendering, RAVEL extracts Dense connectivity, dimensions, numeric
semantics, parameter representation, and coefficient statistics from the
hls4ml graph. The resolved `wide-sequential-v1` plan packs weights at generation
time into one sequential ROM, derives its word width and depth, and emits an
ordered lane-local MAC schedule. Parameter statistics are descriptive and do
not change the generated architecture.

RAVEL renders owned files from the resolved design and read-only parameter
payload through strict templates. The renderer cannot inspect a `ModelGraph`;
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
- Source-conversion consistency compares Keras/HGQ and clean hls4ml in canonical
  fixed-point integer codes; it is not a model-accuracy or convergence test.
- Performance qualification records Vitis HLS measurements without target
  pass/fail limits.

RTL simulation, IP export, implementation timing, and board validation remain
separate activities. Aria 1.5.0 does not promote HLS synthesis into proof of any
of those layers.
