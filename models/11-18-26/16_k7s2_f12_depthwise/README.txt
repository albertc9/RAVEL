16_k7s2_f12_depthwise

New for the converter: 12 x 7, step 2 -> depthwise 7, step 2 (groups = 12, no activation) + 1x1 conv (12 filters)
Layers: conv 12 x 7x1, step 2 (ReLU) -> max-pool 2x1 -> depthwise conv 7x1, step 2, groups 12 (linear) -> 1x1 conv 12 (ReLU) -> max-pool 2x1 -> flatten -> dense 1 (linear)
Input: (256, 4, 1), 1 GSa/s (256 samples = 256 ns)
Weights+biases 1,021; MACs per window 68,208; heaviest layer conv1 42,000 MACs (1.12 x model 01)
1 Hz threshold: 5.606; in 1/16 units 90 (= 5.6250, rounded up). Trigger when score > threshold.
V_eff vs model 01: +21% at 10^16 eV, +13% at 10^18.5 eV
RAVEL (main, 2026-09-17): needs a depthwise conv + 1x1 conv stage
Training: plain; trained run k7s2_dw12_amphp

Files:
  run_16_k7s2_f12_depthwise.keras: main
  verification_data_16_k7s2_f12_depthwise.npz
  ARCHITECTURE.txt, architecture.json: per-layer shapes, weights, MACs, learned bit widths

Verification (1000 windows: 500 neutrino, 500 noise; ../README.txt section 8): noise windows above threshold 0, neutrino windows above threshold 82.
Reload check (each .keras reloaded from disk, scored on X, compared with the stored trained-model scores):
  run_16_k7s2_f12_depthwise.keras: max |diff| 0.00e+00, same trigger decision at 1 Hz 1.0000
Exported with keras 3.11.0, hgq2 0.1.9, torch 2.13.0, numpy 2.2.6.
