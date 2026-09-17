# K5/S3 cross-check fixtures

Built with `make_temporal_model()` in `test_temporal_family.py`, Keras seed 1700,
K5/S3 and non-overlapping two-row Pool after both convolutions:

- `two_block_k5s3_c3.keras`: input (256, 4, 1), filters (3, 3).
- `two_block_k5s3_heldout.keras`: input (192, 4, 1), filters (2, 4).

Saved in independent processes. These are synthetic correctness and scheduling
fixtures; they do not establish classification accuracy. v1.7.2 qualification
records their hashes and validates them through C, CoSim and RTL protocol tests.
