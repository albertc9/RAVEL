# Aria 1.7.1: automatic streaming optimization search

Status: proposed design, 2026-09-15. No implementation or new performance result
is claimed. Inspected source: `devel@5fefb44`. This document scopes the requested
1.7.1 work; it does not change the installed version or publish a release.

## 1. Outcome

For a supported temporal model and its selected P/D configuration, derive legal
implementation candidates from geometry, numeric semantics, and data dependencies;
select a complete streaming implementation using a deterministic bounded search.

The new capability is an implementation-schedule search, not a table mapping
model names or shapes to handwritten Conv2 templates. A shared backend may still
use templates to print C++ syntax. All scheduling, indexing, storage, lane, and
connection decisions must already exist in the immutable selected plan.

P continues to describe external temporal packing and its existing front-stage
implementation. D continues to describe final Dense feature-group parallelism.
Neither becomes a global convolution parallel factor. Existing D4/PHARA coupling
is retained for compatibility. The new search initially varies downstream block
implementations while preserving the selected front and Dense architecture.

## 2. Evidence and architectural gaps

The qualified target has input to Conv2 of 42 x 4 x 12, a 5 x 1 kernel, stride
3 x 1, and 12 output filters. The existing native Conv2 reads 168 position words.
Its loop requests II=1 but achieves II=5, producing stage II=843 and top II=844.
The matched vendor log reports a carried dependence between the window update
and the call that reads that window for convolution arithmetic. The arithmetic
function itself achieves II=1 with latency 2. This motivates schedule changes;
it does not prove that one inline pragma or buffer copy will achieve II=1.

| Existing stage | HLS frame interval, cycles |
|---|---:|
| Front PHARA Conv/ReLU/Pool | 47 |
| Front-to-Conv2 repack | 185 |
| Conv2 | 843 |
| ReLU | 55 |
| Pool2 | 107 |
| Pool-to-Dense repack | 38 |
| Dense | 16 |

These stage intervals are not additive. The exact reports are under
[the 1.7.0 qualification snapshot](../../references/qualification/aria_1_7_0_multi_conv/README.md).
The target's routed resources are 22,581 LUT, 12,472 FF, 0 BRAM tiles, 64 DSP,
with WNS +0.475 ns at a requested 5 ns period.

Current code has useful seams but does not yet implement this search:

- `planning/strategies.py` supplies only the native downstream temporal-block
  implementation. There are no generated channel-mixing alternatives.
- `planning/schedules.py` records output dependencies in accepted input words,
  explicitly not clocks. This is not an achieved-II predictor.
- `planning/chain.py` combines maximum stage cycles, one resource proxy, and
  summed latency. It does not model burst timing, FIFO occupancy, or state hazards.
- Candidate identity currently contains strategy ID/version only; multiple
  schedules of one strategy require an additional canonical plan fingerprint.
- Bridge resolution chooses a local minimum before extending the frontier.
  Different bridge buffering and rate profiles must participate in whole-chain
  selection when they can change downstream feasibility or throughput.
- Unknown cycle estimates currently become infinity for ranking. That cannot
  distinguish several unknown candidates or establish their performance order.
- Source ownership and storage descriptions have strategy-name conditionals;
  new lowerings need explicit plan-owned descriptions, not another implicit
  legacy-template label.

## 3. What is generated

### 3.1 A hardware schedule, not another model graph

hls4ml's ModelGraph remains the semantic authority. Immutable graph facts remain
the input. Introduce a typed schedule representation inside RAVEL's existing
Streaming IR concept, with these contents:

- iteration domains: input positions, valid output windows, kernel taps,
  input channels, output filters, and pooling windows;
- affine index maps, lane-to-logical-index maps, and tail masks;
- read/write dependencies and storage lifetimes;
- arithmetic operations with exact widths, casts, rounding, overflow and order;
- window update, window capture, arithmetic and output handshake events;
- lane allocation, storage ports/banks, FIFO capacity and reset behavior;
- execution phases, initiation constraints, and qualified primitive timing models.

For example, a convolution tap maps to
`input[output_row * stride_h + kernel_row,
       output_col * stride_w + kernel_col, input_channel]`.
The generator derives this mapping for every supported geometry; it must not
contain a special case for height 42, width 4, channels 12, or a model filename.

Separate immutable window values consumed by a computation from mutable line
buffer state. Where copying, bank rotation or forwarding is required, explicitly
represent its lifetime and ports. Storage cannot be reused until its reader has
finished or has captured the values. Backpressure must preserve this invariant.

### 3.2 Transformations and legality

Use a finite versioned transformation grammar over that schedule:

1. Separate window production from arithmetic consumption, introducing a
   snapshot/register or bounded FIFO only when its lifetime can be established.
2. Change function/pipeline placement at legal dependency cuts; generated C++
   boundaries follow the plan rather than accidentally imposing the schedule.
3. Vectorize independent positions along an explicitly named spatial axis.
   Derive useful factors from extents and stream layout, bounded by policy.
4. Partition or bank storage to support the selected accesses and lane count.
5. Derive matching stream packing, ReLU and pooling schedules and bridges.
6. Remove redundant bridges only when complete endpoint contracts are equal.

The initial generator handles channel-mixing arithmetic with full channel/filter
parallelism, as in the current native latency implementation. Channel folding,
arbitrary reduction-tree changes, coefficient-specific Conv2 arithmetic, dead-row
elimination and temporal multi-window vectorization are later transformations.
They must not be silently bundled into position vectorization. Initially, spatial
position factors are legal divisors within a bounded range (including 1 and odd
factors where supported); incomplete groups require a separately tested masked
lane primitive before they enter the search domain.

Each transformation has preconditions, an effect on the schedule, and explicit
legality evidence. It must preserve integer-code semantics, feature order, token
counts, reset and stall behavior. Never disable a dependency warning without a
proof of the relevant read/write independence. Changing accumulation order is
illegal unless the extracted numeric semantics justify it; real-number algebra
alone does not justify fixed-point reassociation or moving casts across ReLU/Pool.

Automatic search needs a finite set of qualified transformations and primitives.
It does not invent arbitrary HLS source or solve unconstrained hardware synthesis.
Flexibility comes from composing model-derived schedules, not adding one template
for each geometry or each lane count.

## 4. Search and evaluation

### 4.1 Deterministic enumeration

The pure search module takes immutable facts, fixed P/D choices, target part/clock,
a versioned policy and a pinned calibration profile. It returns one immutable
selection plus a search report, or structured findings.

1. Retain the exact 1.7.0 composed implementation as the incumbent candidate.
2. Generate candidate schedules by applying legal transformations in a canonical
   order. Deduplicate by full schedule/contract hash, not strategy name.
3. Reject illegal numeric, storage, port, stream and reset combinations early.
4. Connect candidates with all relevant legal bridge variants.
5. Maintain a frontier of complete or partial plans, then evaluate full chains.
6. Select deterministically using the policy below; emit reasons for rejection,
   dominance and selection, including why the incumbent was retained if applicable.

Candidate, transformation-depth, frontier, event and memory bounds are explicit
policy fields. Exceeding a completeness bound returns a structured search-bound
finding; it never silently truncates the domain and claims an optimal result.
The finite domain is itself part of the policy, so even an exhaustive result is
only best within that declared domain, not globally optimal hardware.

Prune only plans with equivalent future-observable contracts and execution state.
Equal word width and scalar cost are insufficient: burst phases, frame restart,
buffered state and backpressure behavior can change downstream costs. Keep those
alternatives until equivalence or safe dominance is established. For uncertain
costs, do not prune by point estimates presented as exact measurements.

### 4.2 Throughput and storage evaluation

Retain input-word dependency traces as functional evidence. Add an analytical
timing evaluator using qualified primitive service intervals/latencies, plus
finite-buffer token execution across consecutive frames. Include fill, drain,
frame restart, bridge transfer, blocking reads/writes and FIFO occupancy.

Compute lower bounds separately: input bandwidth, output bandwidth, work/lane
capacity, storage-port capacity and recurrence constraints. Their maximum is
a lower bound, not a claim that HLS will realize that interval.

A repeated normalized execution state establishes the simulator's steady-state
period. If repetition cannot be established within its explicit bound, report
the result as unknown rather than treating a few frames as proof of steady state.
The simulator predicts a given primitive timing model; it does not prove RTL
performance or liveness under every possible external stall sequence.

Use resource vectors (LUT, FF, DSP, BRAM, storage bits/ports), explicit confidence
and timing risk. Storage port feasibility and target capacity are hard limits
when established; an uncalibrated resource guess cannot certify fit. No hidden
10%-of-device or similar budget is inferred from the present target's usage.

The default policy first enforces legality and qualification domain, then ranks
conservative predicted whole-frame interval, normalized resource cost, latency
and stable plan hash. It contains pinned margins and calibration applicability
rules. Unknown estimates remain explicit and cannot silently displace a qualified
incumbent. A qualified generated candidate needs calibrated coverage or matching
measurement evidence before the release policy promotes it over that incumbent.

### 4.3 Software search and vendor characterization

Ordinary `analyze` and `convert` automatically run deterministic software search.
They do not discover installed vendor tools, contact Ubuntu, or read mutable
historical results to select a different plan. `Vitis.Run` keeps its existing
meaning: build the selected project, not choose a different architecture.

During implementation and qualification, an opt-in experiment runner exports
candidate plans, runs their C checks and HLS synthesis under a bounded job budget,
and routes promising complete candidates. It proposes a versioned calibration
profile for review and inclusion with the release. Vendor execution stays in an
adapter outside the pure search module. An optional result cache may accelerate
this experiment runner only; source/tool/part/clock/reset hashes are mandatory,
and cache contents are not an implicit input to normal conversion.

This scope includes the experimental runner for calibrating and comparing the
new search, but defers a new public per-model vendor-tuning operation. Thus users
get automatic search without managing per-stage knobs or an unexpected synthesis
job on every conversion. Future vendor-driven selection must consume an explicit
frozen evidence input and produce a reproducible selection lock.

## 5. Module design and integration

Keep a deep internal module with a small interface:

`search(facts, constraints, policy, calibration) -> SearchResult`

`SearchResult` contains selected plan, predicted metrics, confidence, diagnostics,
candidate counts and a reproducible decision record. It owns candidate derivation,
dependency legality, schedule composition, costing and deterministic selection.
There is no public method per transformation.

The existing renderer consumes the selected immutable schedule. A generic backend
emitter lowers loops, indexed reads, registers, pipelines, streams and arithmetic
to HLS C++; it cannot make new parallelism or cost decisions while printing code.
Unchanged legacy code continues through the existing adapter. Native alternatives
retain their hls4ml ownership; new schedule-generated code is explicitly RAVEL-owned.

The meaningful variable seam is between pure plan/search and backend execution:
the analytical evaluator and actual vendor runner have different effects and
evidence. Do not introduce public plugins or a large hierarchy of one-line modules.

Expected edits are concentrated in planning, generation policy, schedule lowering,
manifest/architecture identity, report binding and the experiment runner. Frontend
semantics, the clean baseline and parameter extraction remain authoritative.

## 6. Compatibility and reproducibility

The requested 1.7.1 search deliberately expands the earlier fixed-strategy scope.
It needs a new explicit policy identity, for example `aria-stream-search-v1`;
it cannot silently reuse `bounded-temporal-dp@1` to mean a different algorithm.
Earlier notes reserving generalized search for a later generation are superseded
only for this bounded, qualified downstream temporal-block scope.

- New 1.7.1 multi-block conversions use the new policy; existing single-block
  P/D plans retain the previous implementation and regression guarantees.
- Existing project inspection and architecture-preserving refresh replay the
  recorded policy and selected schedule. Refresh never reruns search to change
  lanes, buffers, stage splitting or fusion. Reoptimization uses ordinary conversion.
- New weights invalidate source-bound QoR. Any coefficient-dependent realization
  must remain inside its recorded architecture envelope and be verified again.
- Exact runtime HLS measurements are tied to parameters and source. Generic
  calibration is labeled as an estimate with a stated applicability domain.
- Cache warmth, worker completion order, output path and wall-clock time must
  not affect ordinary selection. Search bounds use deterministic counts rather
  than a wall-clock cutoff.

Record policy, generator and calibration versions/hashes; complete candidate
schedule identity; constraints and search bounds; explored/rejected counts;
selection reasons; predicted metrics/confidence; and source ownership. Keep
measured HLS/RTL/OOC evidence separately bound to the exact source closure.

Plan-affecting data belongs in architecture identity. Auxiliary trace verbosity
and elapsed search duration do not. Version manifest/qualification schemas when
required by the new fields; preserve historical readers and validate all new
records. This design does not preassign schema numbers before that compatibility
work is complete.

## 7. Scope and acceptance

The first qualification domain remains the existing one/two-block temporal family
with its supported numeric contracts and valid geometry. Search support does not
automatically qualify three-block models, padding, arbitrary kernels or new heads.
Unsupported generated schedules leave the explicit native candidate available;
this is recorded candidate selection, not hidden whole-model fallback.

Qualification must demonstrate genuine generalization: the supplied F12 target,
the existing F3 fixture, and new deterministic fixtures varying width, channels,
filters, kernel/stride and pooling tails within the declared support domain.
Include geometries for which different position factors win or no new candidate
is beneficial, and at least one held-out geometry not used to calibrate costs.
Shapes and filenames cannot be selection keys.

The F12 first engineering target is to remove the Conv2 II=5 scheduling bottleneck.
An unchanged narrow bridge suggests an eventual whole-frame interval around
185--200 cycles if Conv2 can approach one position per cycle. This is a hypothesis,
not a promised bound. Further position vectorization must include bridge and Pool
costs; 100 or 50 cycles are not universal acceptance criteria.

Release evidence must show a bit-exact, routable improvement over the matched
1.7.0 F12 incumbent (II 844 at the same 5 ns, part, tools and full-reset settings),
state its measured resource tradeoff, and preserve single-block qualification.
No claim of fourfold improvement appears in release notes without paired evidence.
The software search must actually evaluate multiple distinct generated schedules
and report a geometry-dependent selection; merely changing one template fails scope.

## 8. TDD implementation sequence

Proposed caller-visible test seams are `analyze`, `convert`, `refresh`, project
inspection and qualification records. Schedule/search behavior is exposed through
their selected-plan and diagnostic reports. These extend the existing test surface;
confirm any genuinely new seam before writing tests, as required by the TDD skill.
No tests or product implementation are introduced by this design document.

Implement one failing behavioral test and its minimal implementation per slice:

1. Analysis describes the incumbent bottleneck and distinguishes accepted-word
   dependencies, predicted clock intervals and measured intervals.
2. One model-derived scalar-position schedule converts bit-exactly for two
   differing geometries, preserving casts and output order.
3. Dependency-separated execution passes consecutive-frame, stall and reset
   checks; HLS determines whether the intended II is actually realized.
4. Position factors and matching bridges are derived from facts. Independent
   worked fixtures verify lane indexing, masks and token counts.
5. Whole-chain selection chooses a balanced plan over a locally faster but
   globally blocked alternative, including FIFO/burst constraints.
6. Bounds, unknown costs, deterministic ties and source ownership are observable;
   stale/mismatched evidence cannot qualify a plan.
7. Refresh preserves the selected architecture with changed nonzero weights;
   independent-process regeneration reproduces plan/source hashes.
8. Run the matched target, varied fixtures and existing legacy matrix through
   the required C, RTL and paired hardware gates; freeze release calibration.

Always compare to the untouched clean hls4ml baseline. Keep the built-in corpus
and the supplied 1000-event corpus additive. Preserve per-stage code checks and
fresh-process C execution. RTL tests cover input gaps, output backpressure,
consecutive frames, reset between frames and reset with in-flight state. Resource
and timing evidence comes from actual HLS/OOC reports, not plan estimates.
