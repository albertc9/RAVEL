15_k7s2_f8_depthwise

New for the converter: 8 x 7, step 2 -> depthwise 7, step 2 (groups = 8, no activation) + 1x1 conv (8 filters); heaviest layer 0.75x the current CNN's
Layers: conv 8 x 7x1, step 2 (ReLU) -> max-pool 2x1 -> depthwise conv 7x1, step 2, groups 8 (linear) -> 1x1 conv 8 (ReLU) -> max-pool 2x1 -> flatten -> dense 1 (linear)
Input: (256, 4, 1), 1 GSa/s (256 samples = 256 ns)
Weights+biases 649; MACs per window 41,888; heaviest layer conv1 28,000 MACs (0.75 x model 01)
1 Hz threshold: 5.167; in 1/16 units 83 (= 5.1875, rounded up). Trigger when score > threshold.
V_eff vs model 01: +20% at 10^16 eV, +11% at 10^18.5 eV
RAVEL (main, 2026-09-17): needs a depthwise conv + 1x1 conv stage
Training: plain; trained run k7s2_f8_dw_amphp

Files:
  run_15_k7s2_f8_depthwise.keras: main
  verification_data_15_k7s2_f8_depthwise.npz
  ARCHITECTURE.txt, architecture.json: per-layer shapes, weights, MACs, learned bit widths

Verification (1000 windows: 500 neutrino, 500 noise; ../README.txt section 8): noise windows above threshold 0, neutrino windows above threshold 81.
Reload check (each .keras reloaded from disk, scored on X, compared with the stored trained-model scores):
  run_15_k7s2_f8_depthwise.keras: max |diff| 0.00e+00, same trigger decision at 1 Hz 1.0000
Exported with keras 3.11.0, hgq2 0.1.9, torch 2.13.0, numpy 2.2.6.
