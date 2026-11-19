18_halfrate_lp150_k5s1_depthwise

New for the converter: 0.5 GSa/s input (128 samples) after the 150 MHz digital low-pass; 12 x 5, step 1 -> depthwise 5, step 2 + 1x1 conv (12 filters)
Layers: conv 12 x 5x1, step 1 (ReLU) -> max-pool 2x1 -> depthwise conv 5x1, step 2, groups 12 (linear) -> 1x1 conv 12 (ReLU) -> max-pool 2x1 -> flatten -> dense 1 (linear)
Input: (128, 4, 1), 0.5 GSa/s (128 samples = 256 ns)
Preprocessing before the CNN: 150 MHz low-pass (../lp150_sos.txt), then keep every 2nd sample (../README.txt section 2)
Input gain: 1.862 x the full-band gain, applied after the low-pass (thermal-noise RMS 1)
Weights+biases 973; MACs per window 54,096; heaviest layer conv1 29,760 MACs (0.79 x model 01)
1 Hz threshold: 5.351; in 1/16 units 86 (= 5.3750, rounded up). Trigger when score > threshold.
V_eff vs model 01: +14% at 10^16 eV, +15% at 10^18.5 eV
RAVEL (main, 2026-09-17): needs a depthwise conv + 1x1 conv stage; firmware 150 MHz low-pass + decimation, 128-sample window
Training: plain; input chain includes the 150 MHz low-pass; trained run dec2_k5s1k5s2_dw12_amphp_lp150

Files:
  run_18_halfrate_lp150_k5s1_depthwise.keras: main
  verification_data_18_halfrate_lp150_k5s1_depthwise.npz
  ARCHITECTURE.txt, architecture.json: per-layer shapes, weights, MACs, learned bit widths

Verification (1000 windows: 500 neutrino, 500 noise; ../README.txt section 8): noise windows above threshold 0, neutrino windows above threshold 79.
Reload check (each .keras reloaded from disk, scored on X, compared with the stored trained-model scores):
  run_18_halfrate_lp150_k5s1_depthwise.keras: max |diff| 0.00e+00, same trigger decision at 1 Hz 1.0000
Exported with keras 3.11.0, hgq2 0.1.9, torch 2.13.0, numpy 2.2.6.
