# Aria 1.7.0 multi-block qualification

Implementation source: `8cab26d`. Frozen pre-1.7 baseline:
`afec83563145ea53f3829cdeb30d24d017d6fb2a`. This is a development qualification
snapshot; it creates no release or board-validation claim.

## Environment and inputs

- Vitis HLS and Vivado 2023.2; `xcku5p-ffvb676-2-e`; 5 ns clock request.
- Pinned stack in [aria-reference.txt](../../../constraints/aria-reference.txt).
- Generator development branch `v4.3`, commit
  `170771f22e648b3040535d1cd24a2cf5affe13bc`.
- Target model SHA-256:
  `382bd79b7d9cb78a35d81a32fa2ffb1da07d8e94e4ad93d62dd2eea2585bc0c8`.
- Optional 1000-event NPZ SHA-256:
  `c662edb897f09ea93de1f524b1ce12f00e54b9b028565d4d2082d4c1bb0b64a4`.
- Exact legacy model SHA-256:
  `816a6384509a89808e668747fa5740dab8b3522056df24a8a81da5bbdd57b1ff`.
- The external target model and NPZ are not redistributed. The additional
  synthetic H64/W2/F3/K3/S2 two-block model is frozen in
  [two_block_c3.keras](../../../tests/reference/fixtures/two_block_c3.keras).

[Provenance](provenance.json) binds input hashes and records;
[checksums](checksums.json) cover the exported raw evidence. Reports retain
vendor-generated metadata. Portable generation manifests contain project-relative
source paths.

## Measured comparison

HLS supplies II and latency. Resources and WNS below are routed OOC measurements.
A passing 5 ns route establishes timing at 200 MHz; it is not a separate Fmax
search. BRAM is counted in tiles.

| Model / flow | II | Latency, cycles | LUT | Registers | BRAM | DSP | WNS, ns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target, frozen clean hls4ml, control reset | 3076 | 3091 | 15837 | 12748 | 7 | 0 | +0.800 |
| Target, clean hls4ml, full reset | 3076 | 3091 | 16617 | 14134 | 7 | 0 | +0.928 |
| Target, Aria P8/D4, full reset | 844 | 866 | 22581 | 12472 | 0 | 64 | +0.475 |
| Exact legacy, pre-1.7 P2/D2 | 135 | 140 | 11304 | 7504 | 2 | 40 | +0.673 |
| Exact legacy, Aria 1.7 P2/D2 | 135 | 140 | 11304 | 7504 | 2 | 40 | +0.673 |

The second clean-target run copies the frozen baseline firmware byte for byte
and adds only `config_rtl -reset all`; its binding records that configuration
difference. The original clean baseline is retained. The like-for-like full-reset
comparison reduces target II by 3.64x and latency by 3.57x, with additional LUT/DSP
cost. Legacy II, latency, routed resources, and WNS are identical.

The additional F3 model uses P2/D1 and measures HLS II 94, latency 111 cycles.
It completed C, HLS, CoSim, and protocol qualification; OOC is not required for
this additional fixture and was not run.

## Numerical and protocol evidence

| Project | Built-in C samples | Supplied C samples | Protocol outputs, including replay | RTL CoSim |
|---|---:|---:|---:|---|
| Target | 96 | 1000 | 99 | Pass |
| F3 synthetic | 92 | 0 | 95 | Pass |
| Exact legacy | 74 | 0 | 77 | Pass |

Every compared output agrees in canonical integer codes. Source HGQ to clean C,
clean C to composed C, and clean-baseline words to RTL are separately checked.
The multi-block projects also compare temporary C observation points at unfused
operations, pools, explicit bridges, layout, and Dense. PHARA's fused region is
observed at its exposed endpoint.

XSim 2023.2 protocol tests cover interrupted-inference reset, reset between
epochs, input gaps, output backpressure with stability assertions, consecutive
frames, and exact padded AXI output bits. Pooling drops incomplete windows while
draining the complete frame. Independent generated-bridge C++ tests cover
pack/unpack tails, nonzero input padding, negative scalar bits, and consecutive
calls. Supplied events are additive; the default RTL corpus remains the compact
built-in corpus.

[Independent-process reproduction](reproducibility.json) verifies identical
complete source closure, plan, generation, architecture, structure, and parameter
fingerprints. Host library stamps, archive timestamps, and serialized shared
object IDs are canonicalized without changing parameter bytes or graph semantics.

## Legacy and software gates

- [Legacy comparison](legacy-comparison.json): 19 existing model/P-D combinations;
  identical plans and interfaces and 86 identical firmware files per project.
- All 12 retrained reference models pass current mandatory-corpus C conversion.
  The frozen family matcher agrees with the canonical production parser on the
  existing reference family and the exact external legacy model.
- Full suite: 252 passed. After the final shared-object normalization and fixture
  update, the affected temporal and architecture suite passed 16 tests.
- All five evidence tests pass and check report/vector hashes, manifest/qualification bindings,
  protocol completion, legacy non-regression, and fresh-process reproducibility.
- Generated records are checked against manifest v6 and qualification v5 schemas.
  Existing manifest v1-v5 and qualification v2-v4 readers remain covered.
- Model and parameter-package refresh preserve the recorded architecture;
  coefficient-dependent PHARA logic is regenerated and proven.

## Scope decisions

The target resolves through existing PHARA, explicit lossless bridges, the named
native hls4ml Conv/ReLU/Pool strategy, an identity layout view, and packed Dense.
Native Conv2 measures II 843 and is the bottleneck. The complete candidate is
correct, routable, and substantially faster than the clean baseline, so a custom
channel-convolution optimizer is not required for this scope. No whole-model
vanilla fallback or target-name/extent selector is used.

One and two temporal blocks are qualified. Three or more remain recognizable
but unsupported. Uncalibrated native estimates remain labeled; vendor results
are separate measurements. No universal II<=51 or 5 ns acceptance rule is added.
Accuracy, a tighter-clock Fmax search, multicore scaling, and board integration
remain separate work.

## Reproduce

From a checkout with the pinned environment and vendor settings sourced:

```bash
PYTHONPATH=src KERAS_BACKEND=tensorflow python tools/qualify_aria.py \
  "$TARGET_MODEL" "$OUTPUT_ROOT/composed_target" --inputs "$TARGET_NPZ"

PYTHONPATH=src KERAS_BACKEND=tensorflow python tools/qualify_aria.py \
  tests/reference/fixtures/two_block_c3.keras "$OUTPUT_ROOT/two_block_c3" \
  --temporal-packing 2 --dense-parallelism 1 --skip-ooc

PYTHONPATH=src KERAS_BACKEND=tensorflow python tools/qualify_aria.py \
  "$LEGACY_MODEL" "$OUTPUT_ROOT/legacy_p2d2" \
  --temporal-packing 2 --dense-parallelism 2
```

Each output must be a new project directory. Protocol/OOC directories are
siblings of that project. The driver records source binding before OOC execution
and imports source-bound qualification only after successful verification.
