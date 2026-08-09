# Verification data

Reference datasets are not copied into this repository. Pass a local NumPy
tensor with shape `[samples, 256, 4]` to `generate.py --inputs`. RAVEL records
the tensor shape, dtype, sample count, and SHA-256 of its contiguous bytes in
the generated manifest without recording its source path.
