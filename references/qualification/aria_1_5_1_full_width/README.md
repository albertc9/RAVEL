# Aria 1.5.1 full-width first-convolution evidence

This directory retains the fixed three-model Vitis HLS 2023.2 and RTL CoSim
subset selected for Aria 1.5.1. The release derives a structural first-conv
budget of 140 products per cycle (four output windows times 35 products) rather
than allowing learned zero weights to change the inferred resource limit.

All projects use P4/D2, KU5P, a 5 ns target, 32 deterministic consistency
samples, and Verilog CoSim. Training convergence and application accuracy are
not acceptance criteria.

| Model | Aria 1.5.0 II / latency | Aria 1.5.1 II / latency | First-conv stage interval / loop II | DSP | LUT |
|---|---:|---:|---:|---:|---:|
| `adam_p1_step2` | 94 / 99 | 94 / 99 | 88 / 1 | 8 | 13,149 |
| `adam_hgq_replicate_s2` | 259 / 262 | 94 / 100 | 89 / 1 | 139 | 16,405 |
| `adam_hgq_replicate_s2_300ep` | 344 / 347 | 94 / 100 | 89 / 1 | 8 | 24,964 |

The full-width budget removes the learned-sparsity-dependent first-conv II,
but it is a throughput/resource tradeoff: compared with Aria 1.5.0,
`adam_hgq_replicate_s2` rises from 42 to 139 DSPs and
`adam_hgq_replicate_s2_300ep` rises from 18,518 to 24,964 LUTs. These figures
belong only to the exact generated projects and are not universal resource
gates.

Each model prefix has the immutable generation manifest, schema-v3
qualification record, top-level synthesis XML, first-convolution synthesis
XML, and top-level Verilog CoSim report. `provenance.json` binds those records
to the tracked model archives and exact dependency environment.
