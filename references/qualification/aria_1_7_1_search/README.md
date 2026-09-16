# Aria 1.7.1 analytical search qualification

Vendor-qualified generation: `3a49376`; reviewed implementation: `632090a`.
This is development evidence on `devel`, with no publication or board-validation
claim. Vendor experiments are separate from normal software-only search.

## Environment

- Vitis HLS and Vivado 2023.2; `xcku5p-ffvb676-2-e`; requested period 5 ns.
- Pinned Python stack: [aria-reference.txt](../../../constraints/aria-reference.txt).
- Generator development branch `v4.3` at
  `170771f22e648b3040535d1cd24a2cf5affe13bc`.
- F12 model and additive 1000-event NPZ hashes are in [provenance.json](provenance.json).
- F3 uses the existing [fixture](../../../tests/reference/fixtures/two_block_c3.keras).
- The held-out synthetic model is H80/W3 with F2 then F4, K3/S2,
  [two_block_heldout.keras](../../../tests/reference/fixtures/two_block_heldout.keras).
  It was generated after the calibration examples; it was not added to a model
  whitelist. Its odd final Pool row exercises complete-frame draining.

## Measured results

II and latency come from HLS. Resources and slack come from routed OOC, not HLS
area estimates. BRAM below is in 36-Kib tiles. A 5 ns route proves this target
period; no Fmax search or board result is claimed.

| Model / implementation | II | Latency, cycles | LUT | FF | DSP | BRAM | WNS, ns |
|---|---:|---:|---:|---:|---:|---:|---:|
| F12, Aria 1.7.0 P8/D4 | 844 | 866 | 22581 | 12472 | 64 | 0 | +0.475 |
| F12, Aria 1.7.1 P8/D4, two positions | 124 | 129 | 35579 | 16964 | 64 | 0 | +0.539 |
| F3, Aria 1.7.0 P2/D1 | 94 | 111 | — | — | — | — | — |
| F3, Aria 1.7.1 P2/D1, two positions | 35 | 44 | — | — | — | — | — |
| Held-out H80/W3/F2→F4, one position | 71 | 78 | — | — | — | — | — |
| Legacy single block, Aria 1.7.0 P2/D2 | 135 | 140 | 11304 | 7504 | 40 | 2 | +0.673 |
| Legacy single block, Aria 1.7.1 P2/D2 | 135 | 140 | 11304 | 7504 | 40 | 2 | +0.673 |

F12 II improves by 6.81x against the matched 1.7.0 full-reset project. The cost is
12,998 additional LUT and 4,492 FF; DSP and BRAM remain unchanged. The original
front PHARA and Dense architectures are retained. F3 and the held-out model have
C/HLS/CoSim/protocol evidence; OOC was not run for these auxiliary fixtures.

The selected whole-frame analytical predictions are 139 cycles (F12), 49 (F3)
and 78 (held-out). They remain predictions in the generated manifest even after
separate measured qualification is attached.

## Numerical and stream validation

| Project | Built-in C samples | Supplied C samples | RTL protocol outputs | CoSim |
|---|---:|---:|---:|---|
| F12 | 96 | 1000 | 99 | Pass |
| F3 | 92 | 0 | 95 | Pass |
| Held-out | 92 | 0 | 95 | Pass |
| Legacy single block | 74 | 0 | 77 | Pass |

All canonical codes match. Clean-baseline conversion, transformed C and RTL are
separate checks. Conv/ReLU/Pool/bridge observations validate intermediate codes.
RTL tests include input gaps, output backpressure and stability, consecutive
frames, reset between epochs and reset aborting an active inference. The extra
three protocol outputs are replay events. External NPZ samples augment the
mandatory built-in corpus; they do not replace it.

## Calibration, preservation and records

[calibration.json](calibration.json) records independent development measurements
for one/two-position F3/F12 candidates. Calibration is a bounded geometric and
numeric domain, never a filename lookup. Other legal candidates remain visible
without automatic promotion. Resource envelopes are conservative predictions,
not fitted resource guarantees or a substitute for implementation.

[legacy-comparison.json](legacy-comparison.json) covers 19 existing model/P-D
configurations. All 86 firmware files in each project are byte-identical to the
frozen 1.7.0 evidence. Paired legacy hardware measurements are also identical.

Each project directory contains manifest v7, qualification v5, matched HLS
reports, RTL vectors/protocol logs, and OOC reports when run. [Checksums](checksums.json)
cover the frozen artifacts. Original 1.7.0 comparison reports remain in the
[previous qualification snapshot](../aria_1_7_0_multi_conv/README.md).

Review after vendor qualification changed search diagnostics and error handling,
without changing selected architectures or hardware-generation code. Independent
regeneration checks hardware sources, vectors, vendor scripts and architecture
identities against the qualified projects. The full report-derived host library
stamp can change when diagnostics change; this is reported explicitly rather
than described as identical full-project content.

## Software gates

- Complete software regression: 267 passed after the event-bound repair.
- Additional source-bound release evidence gates validate all four projects,
  paired timing/resource results, corpus coverage and recorded report hashes.
- All four manifests and qualification records pass their published JSON schemas.
- A frozen, source-complete Aria 1.7.0 fixture verifies refresh preserves its
  recorded native architecture even when new conversion selects a window plan.
