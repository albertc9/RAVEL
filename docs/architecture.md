# RAVEL architecture

RAVEL transforms a compatible trained model into a high-throughput FPGA
inference implementation while preserving the model semantics. Training is
outside the RAVEL workflow.

## Aria scope

Aria is the RAVEL 1.x generation. It automates the fixed CNN-Core-Generator
pair-parallel, two-row streaming specialization. It does not perform design
space search, schedule multiple inference contexts, or select replicated IP
cores from a system throughput target.

The primary workflow is a Python API:

1. Create an ordinary hls4ml model graph from a compatible Keras/HGQ model.
2. Call `ravel_hls.optimize_project()` or the
   `ravel_hls.convert_from_keras_model()` convenience API.
3. Generate a clean hls4ml baseline in a staging directory.
4. Build RAVEL semantic and streaming representations.
5. Apply the fixed, legality-checked Aria transformation sequence.
6. Render and validate the specialized hls4ml-compatible project.
7. Atomically publish the project only after enabled checks pass.

RAVEL owns the configuration, specialization, generated specialized sources,
and validation evidence. hls4ml is isolated behind an adapter and remains the
model-integration and generic-baseline provider.

## Transformation model

The Semantic IR records topology, parameters, quantization, and fixed-point
semantics. The Streaming IR records packing, rates, buffers, parallel
allocation, interfaces, and legal fusion. Implementation changes preserve the
Semantic IR contract and express their effects in the Streaming IR.

Aria applies these passes in order:

1. `PackTemporalInput2x`
2. `FuseRepackReshapeIntoFirstConv`
3. `PropagateWideReLUStream`
4. `SpecializeNonOverlappingMaxPool`
5. `StreamFlattenIntoDense`
6. `BindShallowInternalFifos`

Passes are semantic operations with explicit legality checks. They are not
defined as regular-expression edits to generated filenames. RAVEL renders its
managed files once from the final IR using strict Jinja2 templates. Unaffected
project files remain hls4ml-owned.

## Boundaries

Production code lives under `src/ravel_hls`. Independent proof assets live
under `tests`. The complete CNN-for-Arianna consumer workflow lives under
`references/cnn_for_arianna`, while curated retired-generator evidence is
isolated below its `legacy` directory. Neither reference nor legacy code is a
production import dependency.

RAVEL conversion stops at a generated, non-vendor-validated HLS project.
External consumer builds own Vitis execution, RTL/IP export, measured
interface confirmation, and system implementation. RAVEL may import completed
vendor reports later, but conversion never presents source inspection or
historical reports as measured hardware performance.

## Verification layers

Correctness and performance are independent:

- Structural validation checks project shape, configuration, templates, and
  generated-source contracts.
- Transformation equivalence requires bit-exact agreement between the clean
  hls4ml baseline C++ and RAVEL-optimized C++ for identical quantized inputs.
- Model fidelity reports the numerical Keras/HGQ-to-HLS relationship without a
  universal promotion threshold.
- Performance qualification exists only after an explicit vendor-tool flow
  records interface, initiation interval, latency, resources, and timing.

The logical model interface, source-level HLS stream interface, and expected or
measured RTL interface remain separate contracts.
