# Aria 1.7.1: analytical streaming optimization search

Implementation on `devel`; no publication is performed by this work.

## Scope and workflow

1. Extract geometry and numerical contracts from the post-hls4ml ModelGraph.
2. Preserve the requested P/D, front stage and Dense implementation.
3. Retain the native downstream temporal block and generate captured-window
   schedules. Legal position factors divide the width and are at most four.
4. Construct each complete chain, including lossless bridges, ReLU and Pool.
5. Evaluate conservative frame-interval and separate LUT/FF/DSP/BRAM estimates
   at the requested clock. Apply per-core resource ceilings.
6. Select deterministically within calibrated coverage, or retain the incumbent.
7. Lower the recorded schedule and verify canonical codes against a clean
   hls4ml baseline. Record all decisions separately from vendor measurements.

Normal `analyze` and `convert` do not invoke vendor tools to search candidates.
There is no public `tune` operation. Existing explicit build/record operations
remain available. Development qualification uses `tools/qualify_aria.py`.

The release keeps one/two-block recognition and existing single-block behavior.
It does not introduce filter/channel folding, coefficient-specific Conv2
arithmetic, arbitrary graph rewriting, or temporal multi-window vectorization.

## Generated implementation

`WindowSchedule` describes input/output row domains, independent positions,
channels, filters, kernel/stride, pooling and stream-word counts. The registered
`aria-window-stream` strategy derives schedules from facts, without model names
or complete-shape dispatch. The backend registry lowers the selected schedule
using one shared C++ implementation.

Each incoming position captures its window in local fully partitioned values.
History updates do not consume arithmetic results. Ordered, fully unrolled MACs
retain native per-product casts, bias conversion, accumulation order and final
conversion. Pool compares values after the native pool-accumulator conversion.
ReLU and bridges preserve canonical ordering and packing. Incomplete Pool rows
are drained. No false-dependence directive is used.

Row counters restart per inference. History becomes valid only after a complete
fresh kernel window; output cannot consume the preceding frame's history.
Blocking streams preserve state under backpressure, and full-register reset
aborts in-flight work. Existing accepted-input-word dependency traces remain
separate from predicted clock-cycle schedules.

## Selection policy and limits

Policy identity: `aria-stream-search`, version 1. Candidate hashes include full
stage schedules and bridge contracts. Selection orders predicted frame interval,
normalized independent resource envelopes, latency and canonical identity.

The initial pinned cost profile is `ku5p-captured-window-2023.2`, version 1.
Its development evidence is recorded in
[calibration.json](../../references/qualification/aria_1_7_1_search/calibration.json).
Automatic promotion currently requires explicit `xcku5p-ffvb676-2-e` and 5 ns,
position factors 1 or 2, width at most 4, channels/filters at most 12, kernel at
most 5 rows, stride at least 2, non-overlapping two-row Pool, at most 128 input
rows to the downstream block, and the bounded fixed-point types documented by
the profile capability. Other legal schedules remain visible but uncalibrated.
Width 1 has scheduling optimization but no position-parallel speedup.

Frame prediction includes process fill/restart and bridge packing overhead, not
only multiplication throughput. Resource envelopes deliberately reserve separate
LUT/FF/DSP/BRAM budgets; native constant folding can consume considerably less.
These predictions are not achieved II, routed timing or fit guarantees.

Optional `Optimization.ResourceLimits` maps `LUT`, `FF`, `DSP`, and `BRAM` to
nonnegative per-core ceilings. BRAM is counted in 18-Kib blocks. Unspecified
limits use known device capacity, without a system reserve. Unknown estimates
cannot satisfy an explicit ceiling. Resource-constrained search is currently
available for two-block designs; supplying ceilings for a single-block design
returns an unsupported finding rather than silently ignoring them.

```python
config = {
    "HLS": {"Part": "xcku5p-ffvb676-2-e", "ClockPeriod": 5},
    "Optimization": {
        "TemporalPacking": 8,
        "DenseParallelism": 4,
        "ResourceLimits": {"LUT": 160000, "DSP": 1800},
    },
    "Verification": {"Mode": "required"},
}
```

Search evaluates the incumbent first and at most 32 downstream alternatives.
The current generator produces at most five including the incumbent. Bounds are
count-based and deterministic. A bound retains a legal selectable evaluated plan
and reports incomplete exploration; no global optimality is claimed. No feasible
selectable plan yields structured findings, while its search diagnostics remain
available in the analysis report.

## Records, replay and extension points

Manifest schema 7 records the schedule, source ownership, calibration policy,
constraints, predictions, candidate identities and exploration counts. Historical
schemas remain readable. Vendor qualification remains source-bound and separate.

`refresh` checks structure and parameter descriptors, then replays the recorded
architecture through compatible versioned lowerings. It does not rerun current
search defaults. Updated parameters receive new correctness checks and invalidate
old hardware qualification. Ordinary conversion performs reoptimization.

Future strategies can use the existing typed capability registry, immutable
schedule contracts and backend lowering registry. This release deliberately
keeps one implemented window primitive and a small domain; it does not add a
universal optimizer framework. Outputs remain concise and undecorated.

## Verification

TDD covers public analysis, conversion, historical and parameter refresh, project
inspection and source-bound qualification. An actual archived Aria 1.7.0 project
protects replay compatibility. Built-in stimuli always run; supplied vectors are
additive. Qualification covers F12, F3, an unseen width-3 mixed-channel geometry,
and the existing single-block regression family. HLS, RTL protocol and routed
OOC evidence remain distinct from software predictions.
