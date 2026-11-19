12_halfrate_lp150_k5s1_k5s2
New for the converter: 0.5 GSa/s input (128 samples) after a 150 MHz digital low-pass; conv steps 1 and 2.
Trained: dec2_k5s1k5s2_f12_amphp_lp150 (prod_p2_v6, chain amphp_lp150); checkpoint /data/homezvol2/abarlett/ml-trigger-dev/data/models/fLow_0.03-fhigh_0.5-rate_1.0/adam_p2v6_hgq_dec2_k5s1k5s2_f12_amphp_lp150/adam_p2v6_hgq_dec2_k5s1k5s2_f12_amphp_lp150_best.pth
Main version: input (128, 4, 1), 1,477 weights+biases, 113,952 MACs per
256 ns window, heaviest layer conv2 (83,520 MACs). Full layer table and learned bit widths:
ARCHITECTURE.txt (readable) / architecture.json (machine-readable), written from the exported .keras files.

Input convention (all models): the AMPLIFIED trace at 1 GSa/s (series-400 amp; in our simulation followed by
the measured ~100 MHz high-pass), divided by 9.687 uV (the 30-500 MHz thermal-noise RMS of the v6 configs) so that thermal noise has RMS ~1 (verification noise windows: std
0.999). Layout (time, antenna, 1), antennas in station order. Output: one raw score; trigger when score > threshold.
Preprocessing BEFORE the CNN (both versions): a digital 150 MHz low-pass, 4th-order Butterworth IIR at
1 GSa/s (scipy butter(4, 0.150, fs=1.0, output='sos'), coefficients in ../lp150_sos.txt), the same type
as the FPGA high-pass. The CNN runs at 0.5 GSa/s: keep every 2nd sample (samples 0, 2, 4, ...).
  run_12_halfrate_lp150_k5s1_k5s2.keras: input (128, 4, 1) = already decimated (as firmware would do it)
  run_12_halfrate_lp150_k5s1_k5s2__decimate_in_model.keras: input (256, 4, 1); its first layer (a fixed 1x1 conv, step 2,
  weight 1) does the decimation. Use X_1GSa from the verification file for this version.

1 Hz threshold (score units, from 10 x 10^7 noise windows through the same chain): 5.611
  verification noise windows above it: 0 of 500

verification_data_12_halfrate_lp150_k5s1_k5s2.npz: X (main version input), X_1GSa (N, 256, 4, 1) for the 256-sample version, y (1 = neutrino), energy (lg E/eV,
-1 for noise), scores (trained torch model), threshold_1Hz. 500 neutrino windows (v6 V_eff events, enhanced back lobe,
pulse peak inside the window) + 500 thermal-noise windows; the same events/noise draws for every model of the package.
Reload check (each .keras reloaded from disk vs the trained model on the verification windows):
  run_12_halfrate_lp150_k5s1_k5s2.keras: max |diff| 0.00e+00, same trigger decision at 1 Hz 1.0000, OK
  run_12_halfrate_lp150_k5s1_k5s2__decimate_in_model.keras: max |diff| 0.00e+00, same trigger decision at 1 Hz 1.0000, OK
Other trained models with this architecture (not included; available on request):
  dec2_k5s1k5s2_f12_amphp: same layers, trained WITHOUT the low-pass (every 2nd sample only)
Exported with: keras 3.11.0, hgq2 0.1.9, torch 2.13.0, numpy 2.2.6
