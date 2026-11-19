# RAVEL architecture

RAVEL transforms a compatible trained model into a specialized FPGA inference
project while preserving the model semantics. Training remains outside RAVEL.

## Aria workflow

Aria 1.8.0 exposes three Python operations over one configuration:

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

Aria 1.8.0 derives the supertile from the extracted geometry instead of fixing
it. With kernel height `K`, stride `S`, and pool height `P`, one pooled row
spans `K + (P - 1) * S` input rows and advances the input by `P * S` rows. The
row schedule emits at most one pooled row per cycle. When the words read so far
do not cover the next pooled row, the cycle reads one word and produces nothing
(a fill cycle); after the last output the remaining input words are drained.
Buffer depth, total cycles, and fill cycles follow from the geometry. The
height-5, stride-3 geometry keeps the schedule above, its lowering, and its
source bytes. Other geometries use a separate versioned lowering of the same
hybrid arithmetic graph. A height-7, stride-2 convolution on 256 rows produces
62 pooled rows, so its fused region needs at least 62 cycles.

RAVEL renders owned files from the resolved design and read-only parameter
payload through strict templates. The renderer cannot inspect a `ModelGraph`;
unaffected project files remain hls4ml-owned.

The internal built-in generation registry is immutable and closed. Aria 1.8
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

Manifest schema v6 and later separate the parameter-invariant
`architecture_envelope` from the coefficient-dependent `coefficient_realization`. Refresh preserves
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
  for identical inputs. The optimized project is compiled and predicted in a
  fresh process so a baseline shared-library handle cannot mask rendered-source
  changes at the same output path. The default numeric-contract corpus includes
  extrema, alternating values, position-sensitive impulses at leading and
  trailing input rows, and seeded random samples.
- Source-conversion consistency compares Keras/HGQ and clean hls4ml in canonical
  fixed-point integer codes; it is not a model-accuracy or convergence test.
- Performance qualification records top-level and PHARA stage measurements
  from Vitis HLS. A requested record also requires a passing top-level Verilog
  CoSim report.

RTL simulation, IP export, implementation timing, and board validation remain
separate activities. Aria does not promote HLS synthesis into
proof of any of those layers.


## Typed multi-block pipeline

The frontend owns conversion and native symbol bindings; the single-block input
reshape compatibility adapter is isolated. Canonical immutable facts retain
operations, declared ports, tensors, numeric types, attributes, and parameters.
IDs follow declared graph-port traversal, independent of incidental layer names.
The temporal grammar follows edges and reports geometry separately from strategy
legality. Its layout view declares named axes and C-order scalar strides.

Versioned pure stage capabilities enumerate complete endpoint contracts. A
bounded dynamic program retains non-dominated partial chains by endpoint and
specialization status. It prioritizes legality, conservative bottleneck cost,
resource proxy, latency, and stable identity. Unknown cycle confidence sorts
after finite analytical estimates and does not change legality. Candidate and
frontier bounds fail explicitly. Registered lossless bridges participate in
resolution; no vendor measurements or persistent caches are consulted.

The immutable selected plan contains stages, bridges, schedules, control,
source ownership, delegation settings, and policy identities. The clean hls4ml
correctness baseline is separate from selected delegate settings. Composition
uses the unchanged legacy template adapter, top-level wiring, shared-lane
bridges, and packed Dense weights. Every changed or added source has one owning
step/version and before/after hashes. Undeclared mutations fail publication.

Temporary C instrumentation compares canonical codes at available unfused
boundaries, bridge outputs, layout, and Dense. Production sources retain the
ordinary interface. Qualification v5 can additionally bind independent RTL
protocol traces and routed Vivado OOC reports to the exact source and manifest.
Model accuracy and board-level validation remain separate.


## Block and head forms

Since Aria 1.8.0 the temporal grammar parses a block as one convolution with an
optional ReLU and an optional MaxPool, and the head as a Dense with an optional
tail of ReLU and Dense pairs. The support domain is decided from block facts:
the number of pooled blocks, the number of blocks without a pool, and the
number of hidden Dense layers. Stage capabilities receive the operations of a
block by kind, so one captured-window or shared-arithmetic lowering serves a
convolution alone, with ReLU, or with ReLU and MaxPool.

The frontend expresses two source forms equivalently before conversion, because
the pinned hls4ml does not convert them correctly: a plain Keras `ReLU` layer
and a grouped HGQ2 convolution. A grouped or depthwise convolution becomes an
ordinary convolution with a block-diagonal kernel. No grouped operator exists:
the shared-arithmetic graph is built from the coefficient codes and creates no
node for a zero coefficient, so the stage costs only the nonzero products.

A Dense with several outputs is lowered by the `aria-dense-matrix` stage. Each
step reads one input group and updates some of the outputs. Weights come from
one ROM per output lane, and each output accumulates its products in input
order, as the native Dense does. The products of one step are reduced before
the accumulator, so the accumulator loop carries one addition. The number of
outputs per step is enumerated by the search together with the convolution
stages; a throughput target can therefore trade multipliers for steps. Two
outputs of a step share one multiplier per input value when the packed operand
`a * 2^F + b` fits the multiplier ports of the device; both products are
recovered exactly from the one result. The
ReLU and final Dense after a hidden Dense stay native as `hls4ml-head-tail`
and carry one word per inference.

Complete-chain enumeration is bounded by a factor of 64 for each searched stage
after the first. Coverage of the shared-arithmetic cost profile is a versioned
range of measured geometries; a legal candidate outside it stays visible and is
not promoted over a calibrated one.
