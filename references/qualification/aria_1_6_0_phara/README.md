# Aria 1.6.0 PHARA qualification snapshot

This snapshot records the PHARA result selected on commit `e79d3b0`. Release
and board integration remain separate decisions.

## Environment

- Part: `xcku5p-ffvb676-2-e`
- Clock constraint: 5 ns
- Vitis HLS and Vivado: 2023.2
- Default specialization: P8/D4
- Arithmetic realization: hybrid CSD/CSE and DSP

## Canonical model

| Measurement | Result |
|---|---:|
| HLS II | 51 |
| HLS latency | 56 cycles |
| HLS estimated clock | 4.289 ns |
| RTL CoSim | Pass |
| OOC WNS | +0.997 ns |
| OOC TNS | 0 ns |
| OOC BRAM tiles | 2.5 |
| OOC DSP | 79 |
| OOC registers | 3,840 |
| OOC LUTs | 4,930 |

The routed critical path has 3.883 ns data delay and nine logic levels. The
reported WNS is from a complete out-of-context route at 200 MHz.

HLS stage measurements are:

| Stage | II | Latency |
|---|---:|---:|
| PHARA fused Conv/ReLU/MaxPool | 49 | 49 |
| Dense wrapper | 50 | 50 |
| Dense pipeline | 47 | 47 |

## Three-model check

| Model | II | Latency | HLS clock | HLS DSP | HLS LUT | RTL CoSim |
|---|---:|---:|---:|---:|---:|---|
| `adam_p1_step2` | 51 | 55 | 4.289 ns | 64 | 13,295 | Pass |
| `adam_hgq_replicate_s2` | 51 | 56 | 4.289 ns | 80 | 19,108 | Pass |
| `adam_hgq_replicate_s2_300ep` | 52 | 57 | 4.289 ns | 80 | 23,091 | Pass |

All three projects completed C simulation, synthesis, and Verilog RTL CoSim
without HLS errors.

## Numerical evidence

Transformation equivalence and generated correctness checks passed for the
canonical and three additional models. Their PHARA arithmetic graphs have
symbolic modular proofs:

| Model | Modulus | Proof identity |
|---|---:|---|
| canonical | 65,536 | `f50b5d62c132a99d8c6b96253cfe125770ec1bc74a406598f3f70fcadf9e9648` |
| `adam_p1_step2` | 16,384 | `0598a640c32be5ab66bd97e110adb0dc5a5e24eb86ea77f37fa9840963353a28` |
| `adam_hgq_replicate_s2` | 1,048,576 | `3803ab7cb1e4424343b84bed3c63bbbd89c27e9dc3c13ae7751c43be59f54542` |
| `adam_hgq_replicate_s2_300ep` | 2,097,152 | `1ebfff3f6b37878d8ff02b41804dd71bbd668c4d1f1394faea248c341c33c692` |

The symbolic proof covers the coefficient graph at its fixed-point modular
boundary. RTL CoSim covers the generated testbench inputs. Neither result is a
board or full-system timing qualification.
