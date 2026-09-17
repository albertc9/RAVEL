# Aria 1.7.2: shared arithmetic and target-aware search

Aria searches both convolution stages of supported two-block models, reusing
PHARA's constant-arithmetic graph optimizer and modular equivalence proof.
Existing single-block strategy selection and historical architecture replay are
preserved. External temporal packing and Dense parallelism remain caller choices.

## Optional throughput target

```python
config = {
    "HLS": {"Part": "xcku5p-ffvb676-2-e", "ClockPeriod": 5},
    "Optimization": {
        "TemporalPacking": 8,
        "DenseParallelism": 4,
        "TargetII": 85,
    },
}
```

`TargetII` is a positive integer in clock cycles, not latency or frequency.
Omitting it retains throughput-first ranking. With a target, calibrated candidates
meeting it are ranked by predicted LUT, then other resource costs and interval.
If none meets it, the search prioritizes the predicted interval and reports
`predicted-unmet`. An unknown-cost incumbent is reported as unqualified, never
as meeting the target. Resource limits still apply independently.

Neither 85 cycles nor a 20% device-share budget is a product default. Those are
reference-development objectives. Ordinary conversion never launches a vendor
search; measured II, resources and timing remain separate source-bound evidence.

## Shared arithmetic and stage-specific schedules

The optimizer flattens kernel/channel coordinates in native order and derives
constant matrices from the actual coefficient codes. It reuses PHARA's signed
digit decomposition, common-subexpression sharing, bounded DSP product choices
and symbolic modular proof. Balanced reductions are admitted only where the
proof establishes equivalence to the original per-product and accumulation casts.

The first PHARA stage retains its pool-aligned input schedule while considering
new arithmetic realizations. The downstream stage captures windows and uses an
explicit row/column/stride-phase counter, avoiding a wide remainder operator in
the pipelined loop. Finite position-engine reuse preserves the stream word width;
extra arithmetic phases are needed only for complete convolution windows. Every
input word is still read, including rows that do not produce an output. Its arithmetic, ReLU and Pool preserve native cast boundaries.
Histories become valid only after a fresh complete window; blocking reads/writes
preserve the schedule under gaps and backpressure, and reset aborts in-flight work.

Arithmetic requiring fractional product truncation or saturating accumulation is
excluded from this modular rewrite. It retains existing legal implementations.
Sharing arithmetic capability does not require identical hardware schedules.

## Search and calibration boundaries

The bounded search considers complete stage combinations, including bridges and
the Dense head. Reports distinguish numeric legality, calibration coverage,
resource feasibility and measured qualification. Hitting the candidate bound
produces an incomplete-search report, not a global-optimum claim.

The initial constant-matrix profile targets KU5P at 5 ns with four position lanes,
K5/S3 and non-overlapping two-row Pool, up to 12 input channels and filters,
bounded numeric widths and graph depth/fanout. Reuse factors 1 and 2 have offline
calibration. Reuse 4 remains visible but uncalibrated. New first-stage arithmetic
is calibrated at P8; DSP budget 16 is admitted only for paired first-stage graphs
with at most four multiply nodes. Downstream DSP variants remain uncalibrated. Legal variants outside calibrated
coverage stay visible without automatic promotion. Resource estimates are
conservative analytical predictions, not placement or timing certificates.

## Records and parameter refresh

Manifest schema 8 records the selected graph/proof and target-aware policy;
previous manifest versions remain readable. The architecture envelope contains
the arithmetic policy, numeric contract and DSP budget, not coefficient-specific
graph nodes. Refresh regenerates and proves the graph under that recorded
contract while preserving the selected stage schedules. The search report is
marked as the original selection, its II target is not re-estimated, and hardware
qualification must be repeated for the changed source.

## Qualification

Development tools include `tools/qualify_aria.py --target-ii 85` for an explicitly
requested vendor run. The release evidence compares the original four-position
candidate with the selected implementation, validates internal and supplied
corpora separately, and reports single-core routed utilization at 200 MHz.
System integration, replicated cores and board validation are outside this work.


The automatically selected F12 reference core at `TargetII=85` achieves II 63,
latency 71, routed LUT 27,974 (12.89%), FF 19,260 (4.44%), DSP 64 (3.51%), BRAM 0,
and WNS +0.731 ns at 200 MHz. All four resource shares are below 20%. These are
source-bound reference results, not predictions for arbitrary models. See the
[qualification evidence](../../references/qualification/aria_1_7_2_search/README.md).
