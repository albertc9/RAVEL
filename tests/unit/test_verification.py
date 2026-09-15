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


def test_mandatory_corpus_preserves_wide_input_extrema_as_integer_codes():
    from ravel_hls.verification.corpora import prepare_corpora

    tensor = {"id": "input_0:out0", "shape": [4, 2], "numeric_type": {
        "kind": "fixed", "width": 30, "integer": 30, "signed": True,
        "rounding": "RND", "saturation": "SAT_SYM", "saturation_bits": 0}}
    facts = {"operations": [{"id": "input_0", "kind": "input", "outputs": [tensor]}]}
    corpus = prepare_corpora(RavelConfig({"Verification": {"Samples": 8}}), None, facts)[0]
    assert np.all(corpus.inputs[1] == -536870911)
    assert np.all(corpus.inputs[2] == 536870911)
    assert np.all(corpus.inputs >= -536870911)
    assert np.all(corpus.inputs <= 536870911)


def test_sample_request_cannot_remove_mandatory_extrema_or_quantization_neighbors():
    from ravel_hls.verification.corpora import prepare_corpora

    numeric = {"kind": "fixed", "width": 8, "integer": 4, "signed": True,
               "rounding": "RND", "saturation": "SAT_SYM", "saturation_bits": 0}
    tensor = {"id": "input_0:out0", "shape": [4, 2], "numeric_type": numeric}
    facts = {"operations": [
        {"id": "input_0", "kind": "input", "outputs": [tensor]},
        {"id": "relu_0", "kind": "relu", "outputs": [{**tensor, "id": "relu_0:out0", "numeric_type": {**numeric, "integer": 7}}]},
    ]}
    corpus = prepare_corpora(RavelConfig({"Verification": {"Samples": 1}}), None, facts)[0]
    constant_codes = {int(values[0, 0] * 16) for values in corpus.inputs if np.all(values == values[0, 0])}
    assert {-127, 127, -5, -4, -3, 0, 3, 4, 5} <= constant_codes
    assert "seeded_random" in corpus.record["patterns"]
