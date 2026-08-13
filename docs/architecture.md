# RAVEL architecture

RAVEL transforms a compatible trained model into a specialized FPGA inference
project while preserving the model semantics. Training remains outside RAVEL.

## Aria workflow

Aria 1.6.0 exposes three Python operations over one configuration:

1. `analyze(model, config)` converts to a clean hls4ml `ModelGraph`, extracts
   immutable model facts, matches a versioned family, and resolves a design
   without publishing files.
2. `convert(model, output_dir, config)` reuses that same analysis, creates a
   clean baseline in a staging directory, renders from the resolved design and
   ModelGraph parameter payload, verifies it, and atomically publishes it.
3. `refresh(project, model_or_parameters)` requires the recorded architecture
   contract and atomically regenerates the project.
4. RAVEL resolves omitted optimization axes to the versioned P8/D4 PHARA
   default and executes the legality-checked Aria pass sequence.
5. If `Vitis.Run` is true, the published project runs Vitis HLS and records its
   measured report. Conversion never launches the vendor tool by default.

The published project remains useful independently through `Project.open`,
`refresh`, `build`, `record`, and `link`. A failed vendor run does not roll back
or delete the generated sources; it leaves a diagnostic log and does not create
new qualification evidence.

## Transformation model

The hls4ml `ModelGraph` is the authoritative compiler IR. RAVEL projects it into
immutable model facts and executes resolved-design transformations for packing,
rates, buffers, parallel allocation, interfaces, and legal fusion. The PHARA
path records these transformations in order:

1. `pack-temporal-input`
2. `fuse-pool-aligned-conv-relu-maxpool`
3. `stream-flatten-into-dense`
4. `bind-shallow-internal-fifos`
5. `preserve-phara-dataflow-start-propagation`

Explicit P2/P4 configurations retain the separate Conv, ReLU, and MaxPool
streaming stages and the previous seven-pass sequence.

P2, P4, and P8 carry two, four, and eight chronological rows per input word.
The canonical model uses 128-, 256-, and 512-bit input `TDATA`, respectively.
Widths for other applicable models are derived from model facts. Dense x1/x2/x4
consume one, two, or four extracted filter groups per step while retaining the
fixed-point accumulation order.

P2 and P4 fully unroll the first convolution across the extracted output width.
The implementation plan therefore records a structural multiplier budget equal
to `out_width * kernel_height * kernel_width * input_channels * filters` and
targets loop II=1. This budget is an upper bound: Vitis may still fold constant
products into LUT, shift, or add logic. It deliberately does not depend on the
current zero count, so refreshing learned parameters cannot silently reduce the
declared architecture and reintroduce a resource-limited loop II.

Before rendering, RAVEL extracts Dense connectivity, dimensions, numeric
semantics, parameter representation, and coefficient statistics from the
hls4ml graph. The resolved `wide-sequential-v1` plan packs weights at generation
time into one sequential ROM, derives its word width and depth, and emits an
ordered lane-local MAC schedule. Parameter statistics are descriptive and do
not change the generated architecture.

## PHARA specialization

For the qualified height-5, stride-3 convolution and 2x1 MaxPool, PHARA lowers
the two convolution rows consumed by one pool operation as one supertile. Their
union spans eight input rows. Each width lane computes fourteen accumulator
values, applies the original ReLU and MaxPool comparisons, and emits one pooled
word.

The default P8 implementation uses a row-credit scheduler, Dense D4, and a
hybrid constant-arithmetic graph. CSD decomposition and deterministic common
subexpression elimination implement the LUT portion. A bounded set of costly
products uses DSPs. Symbolic propagation must prove the graph equal to the
reference coefficient matrix modulo the original accumulator width before the
renderer accepts it.

The canonical stage-rate model is 32 input cycles, 42 fused-region cycles, and
42 Dense cycles. The 42-cycle value is a structural lower bound, not a promised
HLS II. The measured top-level II is reported separately.

RAVEL renders owned files from the resolved design and read-only parameter
payload through strict templates. The renderer cannot inspect a `ModelGraph`;
unaffected project files remain hls4ml-owned.

The internal built-in generation registry is immutable and closed. Aria 1.6
explicitly composes its operation extractors, family matcher, strategy,
resolver, executed passes, and Vitis/io_stream renderer binding. Matching checks
all declared families and rejects ambiguity; there is no import-time plugin
discovery or registration-order priority.

## Identity and integrity

Generation identity separates the semantic model, generation-affecting
configuration, and implementation. Model parameters and learned quantizer
state participate in semantic identity. Output location, verification choices,
and whether Vitis was invoked do not change the generation fingerprint.
The aggressive-policy identity and resolved P/D values do. Refresh reuses the
recorded resolved values and changes model state without changing architecture.

Manifest schema v5 separates the parameter-invariant `architecture_envelope`
from the coefficient-dependent `coefficient_realization`. Refresh preserves
the envelope, regenerates the arithmetic graph deterministically, and requires
a new modular proof.

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
- Performance qualification records top-level and PHARA stage measurements
  from Vitis HLS. A requested record also requires a passing top-level Verilog
  CoSim report.

RTL simulation, IP export, implementation timing, and board validation remain
separate activities. Aria 1.6.0 does not promote HLS synthesis into proof of any
of those layers.
