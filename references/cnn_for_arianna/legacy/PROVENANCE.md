# Retired generator provenance

The files below are immutable migration evidence from:

- repository: `git@github.com:NuDAQ/CNN-Core-Generator.git`
- source commit: `38ab525467f587a5718c7d7a4c1eff9c9655aed5`
- migrated on: 2026-08-09

`conversion/` preserves the retired conversion program.
`hls4ml_baseline/` preserves the generic generated network core and weights.
`streaming_golden/` preserves only the files that embody the hand-specialized
wide-stream implementation. `reports/` contains a historical Vitis 2023.2 XML
report from that legacy source.

These files are comparison evidence, not runtime dependencies. In particular,
the historical report must not be imported as qualification for a newly
generated RAVEL manifest; source identity and generated-file hashes differ.
