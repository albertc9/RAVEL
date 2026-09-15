import pytest
from ravel_hls.exceptions import VerificationError
import numpy as np

from ravel_hls.config import RavelConfig
from ravel_hls.verification.equivalence import prepare_stimuli


def test_numeric_contract_sweeps_spatial_and_boundary_rows_with_impulses() -> None:
    inputs, record = prepare_stimuli(
        RavelConfig(
            {"Verification": {"Mode": "required", "Samples": 14, "Seed": 7}}
        ),
        None,
        {
            "shape": [32, 4],
            "numeric_type": {
                "kind": "fixed",
                "width": 8,
                "integer": 4,
                "signed": True,
                "rounding": "RND",
                "saturation": "SAT_SYM",
                "saturation_bits": 0,
            },
        },
    )
    codes = np.rint(inputs * 16).astype(np.int64)

    expected_rows = (0, 1, 2, 29, 30, 31)
    expected_values = (1, -1, 1, -1, 1, -1)
    for sample, row, value in zip(codes[4:], expected_rows, expected_values):
        nonzero = np.flatnonzero(sample)
        np.testing.assert_array_equal(nonzero, [row * 4])
        assert sample.reshape(-1)[row * 4] == value
    assert record["patterns"][4:10] == [
        "positive_impulse_row_0",
        "negative_impulse_row_1",
        "positive_impulse_row_2",
        "negative_impulse_row_29",
        "positive_impulse_row_30",
        "negative_impulse_row_31",
    ]
    assert record["patterns"][-1] == "seeded_random"


def test_boundary_verification_rejects_a_permutation_even_when_final_outputs_could_match():
    from ravel_hls.verification.boundaries import compare_boundaries

    reference = [("max_pool2d_0:out0", np.array([[1, 2, 3]], dtype=np.uint64))]
    with pytest.raises(VerificationError, match="max_pool2d_0:out0"):
        compare_boundaries(reference, [("max_pool2d_0:out0", np.array([[2, 1, 3]], dtype=np.uint64))])
