Load any model: import hgq; import keras; model = keras.saving.load_model('<folder>/run_<folder>.keras')
(keras >= 3.11 with HGQ2; exported with keras 3.11.0, hgq2 0.1.9, torch 2.13.0, numpy 2.2.6).
INDEX.txt lists all models. Each folder has README.txt (what is new, input format, 1 Hz threshold, reload check),
ARCHITECTURE.txt / architecture.json (layers, shapes, weights, MACs, learned bit widths, generated from the .keras
files) and verification_data_*.npz (1000 windows with the trained model's scores).
Input convention (all models): the AMPLIFIED trace at 1 GSa/s (series-400 amp; in our simulation followed by
the measured ~100 MHz high-pass), divided by 9.687 uV (the 30-500 MHz thermal-noise RMS of the v6 configs) so that thermal noise has RMS ~1 (verification noise windows: std
0.999). Layout (time, antenna, 1), antennas in station order. Output: one raw score; trigger when score > threshold.
Model code: hgq_phase2_models_ablation.py (14: hgq_phase2_models_v2.py); keras graphs: fpga_export_lib.py.
