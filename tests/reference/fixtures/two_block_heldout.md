# Held-out temporal model

`two_block_heldout.keras` uses deterministic seed 1700, input H80/W3/C1,
K3/S2 convolution blocks with 2 then 4 filters, non-overlapping two-row Pool,
and a one-output Dense head. It uses the same HGQ2 quantization configuration
as the existing fixture generator in `test_temporal_family.py`.

The model was created after the F3/F12 calibration runs. It validates numerical
and stream generalization across a new width, frame size and channel pairing;
it is not a trained model-quality benchmark. HLS/CoSim/protocol evidence is
recorded under `references/qualification/aria_1_7_1_search/heldout`.
