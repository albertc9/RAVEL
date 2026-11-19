08_three_conv_layers
New for the converter: three conv layers (8/8/12 filters), pooling after layers 1 and 3 only.
Trained: light3cv_f8f8f12_amphp (prod_p2_v6, chain amphp); checkpoint /data/homezvol2/abarlett/ml-trigger-dev/data/models/fLow_0.03-fhigh_0.5-rate_1.0/adam_p2v6_hgq_light3cv_f8f8f12_amphp/adam_p2v6_hgq_light3cv_f8f8f12_amphp_best.pth
Main version: input (256, 4, 1), 837 weights+biases, 59,424 MACs per
256 ns window, heaviest layer conv2 (30,720 MACs). Full layer table and learned bit widths:
ARCHITECTURE.txt (readable) / architecture.json (machine-readable), written from the exported .keras files.

Input convention (all models): the AMPLIFIED trace at 1 GSa/s (series-400 amp; in our simulation followed by
the measured ~100 MHz high-pass), divided by 9.687 uV (the 30-500 MHz thermal-noise RMS of the v6 configs) so that thermal noise has RMS ~1 (verification noise windows: std
0.999). Layout (time, antenna, 1), antennas in station order. Output: one raw score; trigger when score > threshold.

1 Hz threshold (score units, from 10 x 10^7 noise windows through the same chain): 5.318
  verification noise windows above it: 0 of 500

verification_data_08_three_conv_layers.npz: X (main version input), y (1 = neutrino), energy (lg E/eV,
-1 for noise), scores (trained torch model), threshold_1Hz. 500 neutrino windows (v6 V_eff events, enhanced back lobe,
pulse peak inside the window) + 500 thermal-noise windows; the same events/noise draws for every model of the package.
Reload check (each .keras reloaded from disk vs the trained model on the verification windows):
  run_08_three_conv_layers.keras: max |diff| 0.00e+00, same trigger decision at 1 Hz 1.0000, OK
Exported with: keras 3.11.0, hgq2 0.1.9, torch 2.13.0, numpy 2.2.6
