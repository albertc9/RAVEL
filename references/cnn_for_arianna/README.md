# CNN for Arianna reference

This directory is the executable Aria 1.0 reference consumer. It owns the
canonical trained Keras/HGQ2 model and invokes only RAVEL's public API. RAVEL
production modules do not import this directory or the retired generator under
`legacy/`.

Use a clean Python 3.11 virtual environment. Do not install the old `HGQ`
distribution alongside `hgq2`: both provide the `hgq` import namespace.

```bash
python -m pip install -c constraints/aria-reference.txt -e '.[reference]'
ravel-hls doctor --json
python references/cnn_for_arianna/generate.py
```

The default run uses required bit-exact baseline/optimized C++ verification,
32 deterministic synthetic samples, the `xcku5p-ffvb676-2-e` part, and a 5 ns
clock target. Use `--inputs test_vectors.npy` to supply a tensor of shape
`[samples, 256, 4]`. Generated output is ignored below `generated/`.

Vitis synthesis is deliberately separate. After running the selected Vitis
2023.2 flow, attach its reports with `ravel_hls.import_vitis_reports`; the
importer checks the immutable manifest and expected RTL widths before recording
measurements.
