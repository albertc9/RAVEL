This zip: models 15-18. Models 01-14 (package 2026-11-08, same files) are in fpga_v6_models_all_2026-11-18.zip.

CNN trigger models for FPGA conversion, 2026-11-18. 4 models, one per architecture.

1. QUICK START
  Load: import hgq, keras; model = keras.saving.load_model('<folder>/run_<folder>.keras')
        (keras 3.11.0, hgq2 0.1.9; exported with torch 2.13.0, numpy 2.2.6). Layer classes: hgq QConv2D, QDense; keras MaxPooling2D, Flatten, ReLU, Reshape (11), Conv2D
        (decimate_in_model versions only).
  INDEX.txt: ranking by V_eff gain and heaviest layer, RAVEL status, size, 1 Hz threshold (also in 1/16 units).
  Top of the ranking: 17, 15, 16, 18. Best that RAVEL main converts as exported: 03, then 05 (3.02 x heaviest layer).
  Each folder: README.txt (model facts), run_<folder>.keras (main version), ARCHITECTURE.txt / architecture.json,
  verification_data_<folder>.npz; alternative versions in 09, 11, 12, 13 (section 8).


2. INPUT AND OUTPUT
  Signal: the AMPLIFIED trace of 4 LPDA channels in station order (series-400 amp; in the simulation followed by the
    measured ~100 MHz high-pass). Layout (time, antenna, 1).
  Scaling: divided by 9.687 uV (the 30-500 MHz thermal-noise RMS of the simulation), so thermal noise at the CNN input
    has RMS 1 (verification noise windows: std 0.999 full band, 0.998 after the low-pass).
  Full-band models: 256 samples at 1 GSa/s = one 256 ns window.
  Half-rate models (12, 13, 18): 128 samples at 0.5 GSa/s = one 256 ns window. Preprocessing, in order, on the
    continuous 1 GSa/s stream of each channel (after the high-pass):
    1. 150 MHz low-pass (12 and 18; not 13): 4th-order Butterworth IIR, fs = 1 GSa/s, -3 dB at 150 MHz, the same
       filter type as the FPGA digital high-pass; 2 second-order sections applied in series
       (scipy.signal.butter(4, 0.150, fs=1.0, output='sos'); scipy.signal.sosfilt). Coefficients (also in
       lp150_sos.txt), one row per section, b0 b1 b2 a0 a1 a2:
       +1.856301062690e-02 +3.712602125379e-02 +1.856301062690e-02 +1.000000000000e+00 -6.727409111915e-01 +1.445351998331e-01
       +1.000000000000e+00 +2.000000000000e+00 +1.000000000000e+00 +1.000000000000e+00 -8.976579400366e-01 +5.271869046316e-01
    2. Gain (12 and 18): the low-pass lowers the thermal-noise RMS to 0.537 x, so the CNN input gain is 1.862 x
       the full-band gain (noise RMS 1 at the CNN input). Model 13: full-band gain.
    3. Decimation: keep every 2nd sample (samples 0, 2, 4, ... of the 1 GSa/s window) -> 128 samples.
    Main version input (128, 4, 1) = after steps 1-3; decimate_in_model version input (256, 4, 1) = after steps 1-2.
  Output: one raw score per window (linear output layer); trigger when score > threshold.
  1 Hz threshold: noise trigger fraction 2.56e-7 per window = one decision per 256 ns on back-to-back windows; from
    10 x 10^7 thermal-noise windows (10 files of 10^7) through the model's own input chain, best checkpoint. INDEX also
    gives it in 1/16 units, rounded up (rate <= 1 Hz).


3. LAYER CONVENTIONS AND QUANTIZATION
  Conv kernels (k, 1): along time only, the same weights for all 4 antennas (exceptions: 10 has a (1, 4) kernel that
    combines the antennas; 11 takes the antennas as input channels).
  Padding valid. Max-pool (2, 1) with step (2, 1) along time, remainder dropped.
  Activations: ReLU after every conv except a depthwise conv (linear); output dense linear; hidden dense + ReLU.
  Depthwise conv: grouped conv with groups = channels (one kernel per channel), followed by a 1x1 conv.
  Flatten order: (time, antenna, channel), row-major; dense weight row index = (t * 4 + antenna) * channels + channel.
  Quantization: HGQ2 fixed point (kif, round RND, overflow SAT_SYM), one learned bit width per tensor (input, kernel,
    bias of every layer); listed in ARCHITECTURE.txt as ap_fixed<W,I>; the .keras file is authoritative.
  First-layer input width learned (conv1 iq): ap_fixed<10,5> for all models except 17 (ap_fixed<11,5>, one more
    fractional bit) and 14 (ap_fixed<12,6>).


4. HOW THE MODELS WERE TRAINED AND EVALUATED
  Training data: simulated neutrino signals (prod_p2_v6: LPDAs at 3 m; band kept down to 30 MHz; back-lobe arrival
    directions enhanced x2 at 90-124 deg, x3 above 124 deg) mixed into thermal noise generated during training; the amp
    + measured high-pass applied to the mixed stream (12, 18: + low-pass and decimation, in float64; no fixed-point
    version of the filter was simulated).
  Training: hinge loss, Adam (learning rate 1e-4, batch 100), 10 epochs, HGQ2 quantization-aware training (start 8
    bits, 4 integer; learned widths); the epoch with the lowest validation loss is exported. Model 17: + distillation
    from model 14 (its README).
  V_eff: simulated neutrino events (prod_p2_v6 V_eff set, back-lobe enhanced), each trigger at its own 1 Hz threshold,
    the same events for every model. INDEX: change vs model 01 (average of its 3 training runs) at 10^16 and 10^18.5 eV.
    Training randomness: 3 runs of model 01 differ by 2-7%. Model 17's value is from 1 run (its 3 plain-trained runs:
    +26% -> +15%).
  Narrow-band chain, measured the same way: the low-pass alone at full rate (model 01 layers, no decimation) +4% ->
    +10% vs model 01; decimation without the low-pass (model 13) -25% -> -16%.


5. CONNECTING TO YOUR CODE
  Read 2026-11-18 from the main branches: CNN-Core-Generator 7e26d15 (2026-10-13), RAVEL 7792367 (2026-09-17),
  AI-Trigger-System 5d4db66 (2026-11-03), CNN-Core-Wrapper 4250135 (2026-10-13).

  CNN-Core-Generator, generator.py
    MODEL_DIR: a folder of this package. Model file: run_<folder>.keras (in place of run_v6_2cv_k5s3_f12_amphp.keras,
      which is model 01). Verification file: verification_data_<folder>.npz; it has the keys generator.py reads (X with
      shape [N, H, 4, 1], y, scores) plus energy, threshold_1Hz and (12, 13) X_1GSa.
    TRIGGER_THRESHOLD: threshold_1Hz of the model (4.953 in the file = model 01).
    np.testing.assert_array_equal(keras_scores, reference_scores): holds for every main .keras (reload check, section 8).
    custom_objects {QConv2D, QDense}: the only HGQ layer classes in these models.
    convert(...) TemporalPacking 8: divides both input heights (256 and 128).
  RAVEL, src/ravel_hls
    domain/temporal.py, recognize_temporal_chain: the accepted model form (section 6).
    generations/aria/matching.py: quantizers must be tensor-wide kif / RND / SAT_SYM; all models here are.
    frontend/extraction.py, _semantic_kind: maps Conv2D and PointwiseConv2D to conv2d; no grouped (depthwise) conv.
  AI-Trigger-System, HDL/rtl
    AI_TRIGGER_PKG.vhd: N_BATCHES 64 x N_BATCH_S 4 = 256-sample chunks, back to back (the window of the 1 Hz
      thresholds); N_LANES 2; CNN_SCORE_FRAC_BITS 10 (score ap_fixed<21,11>); CNN_THRESH_FRAC_BITS 4 -> threshold in
      1/16 units (INDEX column "x16 up").
    CNN_INPUT_CALIBRATION_PKG.vhd: native CNN input signed 10 bits with 5 fractional bits = ap_fixed<10,5>, the
      learned conv1 input width of every model except 17 and 14 (section 3).
    CNN_INPUT_GAIN_PKG.vhd, CNN_INPUT_GAIN_PPM: per-channel CNN input gain; for 12 and 18 the gain after the low-pass
      is 1.862 x the full-band gain.
    No low-pass or decimation stage exists in HDL/rtl.
  CNN-Core-Wrapper, hw/rtl/cnn_core_wrapper_top.v
    WRAPPER_TOP: NUM_TIMESTEPS 256, NUM_CHANNELS 4, INPUT_WIDTH 512 (8 timesteps per beat); 12, 13 and 18 take 128.


6. WHAT RAVEL NEEDS FOR THE OTHER MODELS
  RAVEL main accepts Input -> (Conv2D -> ReLU -> MaxPool2D) repeated -> Flatten -> Dense with 1 output.
  Models 01-06, 12 and 13 have this form. Missing per model:
    15: needs a depthwise conv + 1x1 conv stage
    16: needs a depthwise conv + 1x1 conv stage
    17: needs a depthwise conv + 1x1 conv stage and a hidden dense layer
    18: needs a depthwise conv + 1x1 conv stage; firmware 150 MHz low-pass + decimation, 128-sample window

  Depthwise + 1x1 stage (models 15, 16, 17, 18). After conv -> ReLU -> max-pool:
    depthwise conv: y[t,a,c] = b[c] + sum_k w[k,c] * x[S*t + k, a, c]
                    one kernel per channel (groups = channels), kernel K and step S along time, no activation;
                    y is quantized to its own learned width (input quantizer of the 1x1 conv).
    1x1 conv:       z[t,a,o] = ReLU(b'[o] + sum_c w'[c,o] * y[t,a,c])
    then max-pool (2, 1) step (2, 1).
  15_k7s2_f8_depthwise               in (62, 4, 8), kernel 7, step 2, 8 -> 8 channels: depthwise 6,272 + 1x1 7,168 = 13,440 MACs (ordinary 8 -> 8 conv: 50,176); weights+biases 64 + 72
  16_k7s2_f12_depthwise              in (62, 4, 12), kernel 7, step 2, 12 -> 12 channels: depthwise 9,408 + 1x1 16,128 = 25,536 MACs (ordinary 12 -> 12 conv: 112,896); weights+biases 96 + 156
  17_k7s2_depthwise_dense32          in (62, 4, 12), kernel 7, step 2, 12 -> 16 channels: depthwise 9,408 + 1x1 21,504 = 30,912 MACs (ordinary 12 -> 16 conv: 150,528); weights+biases 96 + 208
  18_halfrate_lp150_k5s1_depthwise   in (62, 4, 12), kernel 5, step 2, 12 -> 12 channels: depthwise 6,960 + 1x1 16,704 = 23,664 MACs (ordinary 12 -> 12 conv: 83,520); weights+biases 72 + 156

  Hidden dense (models 17): flatten -> dense N -> ReLU -> dense 1.
    Model 17: 896 -> 32 -> 1; dense weights+biases 28,704 + 33 (98.6% of the model's 29,137); 28,672 + 32 MACs per
    window; kernel width ap_fixed<11,5> (28,672 x 11 bits = 315 kbit). Each dense weight is used once per window.


7. SUGGESTIONS (ours; not tested in firmware)
  Suggestion: build model 17's hidden dense time-shared, with the weights read from memory. A fully parallel dense
    needs 28,672 multipliers. Per lane at 200 MHz with 2 lanes there are 102 clock cycles per window, i.e. at least 281
    multiplies per cycle (its conv1: 42,000 MACs -> 412 per cycle).
  Suggestion: if only one new firmware stage is added for the narrow-band models, add the 150 MHz low-pass first. It
    gave +4% -> +10% on its own (section 4), and decimation is only usable after it.


8. FILES AND VERIFICATION DATA
  Per folder: run_<folder>.keras (main version); README.txt; ARCHITECTURE.txt / architecture.json (per-layer shapes,
    weights, MACs, learned bit widths, written from the exported .keras files); verification_data_<folder>.npz.
  Alternative versions (identical scores): 09 __dense_equivalent (depthwise conv as an ordinary conv with a
    block-diagonal kernel, off-diagonal weights 0); 11 __channels_input (input (256, 1, 4) directly); 12, 13
    __decimate_in_model (input (256, 4, 1) after steps 1-2 of section 2; a fixed 1x1 conv with step 2 and weight 1
    keeps every 2nd sample; use X_1GSa).
  verification_data_<folder>.npz: X (main-version input), X_1GSa (12, 13), y (1 = neutrino, 0 = noise), energy (lg E/eV;
    -1 for noise), scores (trained model, float32), threshold_1Hz. 500 neutrino windows: V_eff events of all flavours,
    energies and interactions, the 256-sample window placed at a random position containing the pulse peak; 500
    thermal-noise windows from the training-noise recipe through the same chain. The same events and noise draws for
    every model.
  Reload check: every .keras was reloaded from disk and scored on X; all reproduce the stored scores exactly (max |diff|
    0, same trigger decision at 1 Hz 1.0000); per-file values in each README.
  Other files: lp150_sos.txt (low-pass coefficients); model code hgq_phase2_models_ablation.py; fpga_export_lib.py
    (keras graph construction).
