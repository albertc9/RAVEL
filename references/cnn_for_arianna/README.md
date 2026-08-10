# CNN for Arianna reference

This directory is the executable Aria 1.1.0 reference consumer. It owns the
canonical trained Keras/HGQ2 model and uses only RAVEL's public API.

```bash
python -m pip install -c constraints/aria-reference.txt -e .
ravel-hls doctor --json
python references/cnn_for_arianna/generate.py
```

The default run performs required bit-exact baseline/optimized C++ verification
with 32 deterministic synthetic samples, targets `xcku5p-ffvb676-2-e` at 5 ns,
and publishes below `generated/`. Supply `--inputs test_vectors.npy` for a local
tensor shaped `[samples, 256, 4]`; its source path is not recorded.

Vitis is off by default. Add `--vitis` to set `Vitis.Run` true in the same Python
configuration. RAVEL then runs the standalone Vitis HLS 2023.2 launcher after
publication, using the default reset+synthesis stage profile, and automatically
writes `ravel_qualification.json` on success. No user-written conditional or
manual report-import step is required.

The qualification records the measured result for that exact project. It does
not require II 178 or any other application-specific performance number. Enable
additional `Vitis.Stages` only when their separate CSim, CoSim, validation,
export, or Vivado-synthesis evidence is wanted.
