17_k7s2_depthwise_dense32

New for the converter: 12 x 7, step 2 -> depthwise 7, step 2 + 1x1 conv to 16 filters -> Dense(32) + ReLU -> output; this checkpoint was trained to also follow the large CNN's scores (distillation), our best FPGA-sized model so far
Layers: conv 12 x 7x1, step 2 (ReLU) -> max-pool 2x1 -> depthwise conv 7x1, step 2, groups 12 (linear) -> 1x1 conv 16 (ReLU) -> max-pool 2x1 -> flatten -> dense 32 (linear) -> ReLU -> dense 1 (linear)
Input: (256, 4, 1), 1 GSa/s (256 samples = 256 ns)
Weights+biases 29,137; MACs per window 101,616; heaviest layer conv1 42,000 MACs (1.12 x model 01)
1 Hz threshold: 9.169; in 1/16 units 147 (= 9.1875, rounded up). Trigger when score > threshold.
V_eff vs model 01: +51% at 10^16 eV, +16% at 10^18.5 eV (1 training run; the 3 plain-trained runs of these layers: +26% -> +15%)
RAVEL (main, 2026-09-17): needs a depthwise conv + 1x1 conv stage and a hidden dense layer
Training: distillation: loss = hinge + 0.1 x mean((score - clip(score of model 14, +-12))^2); trained run swing_s2_amphp_distill_c12

Files:
  run_17_k7s2_depthwise_dense32.keras: main
  verification_data_17_k7s2_depthwise_dense32.npz
  ARCHITECTURE.txt, architecture.json: per-layer shapes, weights, MACs, learned bit widths

Verification (1000 windows: 500 neutrino, 500 noise; ../README.txt section 8): noise windows above threshold 0, neutrino windows above threshold 86.
Reload check (each .keras reloaded from disk, scored on X, compared with the stored trained-model scores):
  run_17_k7s2_depthwise_dense32.keras: max |diff| 0.00e+00, same trigger decision at 1 Hz 1.0000
Exported with keras 3.11.0, hgq2 0.1.9, torch 2.13.0, numpy 2.2.6.
Available on request: 7 more runs with these layers (3 plain-trained, 4 other training methods).
