# Aria 1.5 retrained-model RTL evidence

This directory retains the fixed three-model RTL CoSim subset selected for the
Aria 1.5.0 release. These models are development/consistency fixtures; training
convergence and application accuracy are not acceptance criteria.

Each prefix has the immutable RAVEL generation manifest, the post-import
qualification record, the original Vitis HLS 2023.2 synthesis XML, and the
top-level Verilog CoSim report. `provenance.json` binds those files to the
tracked model archive and exact dependency environment. The other nine
retrained models remain covered by analysis plus baseline/optimized C++
consistency tests.

Measured II and latency vary with learned numeric types and belong only to the
exact generated project. They are evidence, not universal performance gates.
