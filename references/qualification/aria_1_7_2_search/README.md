# Aria 1.7.2 optimization qualification

Implementation: `409da03` on `devel`. No release or publication was performed.
All vendor runs use Vitis HLS/Vivado 2023.2, KU5P `xcku5p-ffvb676-2-e`, 5 ns.
`final_*` projects use ordinary automatic selection with external P8/D4 and
`Optimization.TargetII: 85`; development candidate selection is never a product
vendor-search loop.

## Matched single-core reference

| Implementation | II | Latency | LUT | FF | DSP | BRAM tiles | Routed WNS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing four-position candidate (`baseline_p4`) | 82 | 85 | 48,507 (22.36%) | 24,078 (5.55%) | 64 (3.51%) | 0 | +0.308 ns |
| Unshared hybrid candidate (`selected_f12`) | 49 | 57 | 39,881 (18.38%) | 26,321 (6.07%) | 64 (3.51%) | 0 | +0.432 ns |
| Automatic target85 selection (`final_f12`) | 63 | 71 | 27,974 (12.89%) | 19,260 (4.44%) | 64 (3.51%) | 0 | +0.731 ns |

The target-aware choice reduces LUT by 42.3% against the matched four-position
baseline and meets II<=85. Each routed resource is strictly below 20%, independently;
TNS is 0 at 200 MHz. Capacities are 216,960 LUT, 433,920 FF, 1,824 DSP and 480 BRAM36
tiles (or 960 BRAM18 blocks). No additional resource class was introduced.
These are this reference's acceptance limits, not product-default budgets.
The baseline is the previously experimental four-position candidate, not a claim
about v1.7.1's default two-position selection.

The unshared candidate demonstrates the throughput/resource tradeoff, not a
claim of a global optimum. Omitting TargetII retains throughput-first ranking;
its resource tie-break can choose a different arithmetic graph.

## Coverage and boundaries

- F12 canonical C verification: mandatory 96-event corpus plus additive 1,000
  supplied events; intermediate boundaries also pass.
- Final C3 fixture: II 61, latency 68; held-out fixture: II 46, latency 53.
- All final projects pass RTL CoSim and the protocol test: continuous frames,
  input gaps, output backpressure, reset between frames and in-flight abort.
  Protocol tests complete 99/99/101 outputs for F12/C3/held-out. RTL uses the
  deterministic built-in corpus; the supplied 1,000 events add C coverage.
- Routed OOC applies to the F12 core. Small fixtures have HLS/CoSim/protocol
  evidence; they do not claim routed timing. No board or multi-core validation.
- Nineteen legacy configurations generate exactly the same 86 firmware source
  files as v1.7.1. Historical v1.7.0/v1.7.1 refresh and parameter-package refresh
  are covered separately by public-API tests.

## Search experiments

`calibration.json` records raw top-level HLS report hashes and distinct sources
for pure CSD/CSE, bounded DSP mapping, uniform reuse and active-window reuse.
Its version 2 policy uses extra phases only for complete convolution windows.
Uniform reuse 2 reached II 92/LUT 28,025; active-window reuse 2 reaches II 63/LUT 27,974.
Every input word is still consumed. Reuse 4 is implemented and exercised, but is
outside automatic calibration coverage. Downstream DSP variants also remain
uncalibrated; first-stage DSP coverage is bounded to four multiply nodes.

`active_*`, `hybrid_*`, `reuse*` and `selected_f12` are development comparisons;
they retain their original manifest/version metadata. Only `final_*` is the
final automatic v1.7.2 acceptance set. Small pure/hybrid probes can deduplicate
to the same graph when the requested DSP budget offers no distinct realization.

## Reproduction

Use `tools/qualify_aria.py` with the model path, a new output directory,
`--target-ii 85`, and `--inputs` for the F12 NPZ. Add `--skip-ooc` for small
fixtures. Source models, supplied corpus and exact implementation revision are
identified in `provenance.json`; the external F12 assets remain in the Generator
repository. Test fixtures are in `tests/reference/fixtures/two_block_k5s3*.keras`.

Each project contains its original manifest, qualification record, HLS reports,
protocol testbench/vectors/log and applicable OOC reports. Reports bind to the
manifest and source closure. `checksums.json` covers the frozen artifacts.
`reproducibility.json` compares all declared compilation inputs and selected
candidate IDs across separate Python processes. Randomly named host `.so`
simulation build products are excluded from source comparison.
