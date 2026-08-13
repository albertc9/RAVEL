from ravel_hls.analysis.phara import (
    AffineGraph,
    AffineNode,
    AffineProof,
    analyze_direct_supertile,
    evaluate,
    prove_equivalent,
)


def test_phara_proves_a_modular_affine_graph_against_known_coefficients() -> None:
    graph = AffineGraph(
        input_ids=("x0", "x1"),
        nodes=(
            AffineNode(id="twice_x0", operation="shift", inputs=("x0",), value=1),
            AffineNode(
                id="result",
                operation="subtract",
                inputs=("twice_x0", "x1"),
            ),
        ),
        output_ids=("result",),
        modulus=16,
    )

    proof = prove_equivalent(graph, reference_coefficients=((2, -1),))

    assert proof.status == "proven"


def test_phara_proof_has_a_stable_canonical_identity() -> None:
    graph = AffineGraph(
        input_ids=("x0", "x1"),
        nodes=(
            AffineNode(id="twice_x0", operation="shift", inputs=("x0",), value=1),
            AffineNode(
                id="result",
                operation="subtract",
                inputs=("twice_x0", "x1"),
            ),
        ),
        output_ids=("result",),
        modulus=16,
    )

    proof = prove_equivalent(graph, reference_coefficients=((2, -1),))

    assert proof == AffineProof(
        status="proven",
        modulus=16,
        reference_coefficients=((2, 15),),
        realized_coefficients=((2, 15),),
        graph_sha256="29477c026555099718e5d6bed3945f5cd86b67417bbd2be12181b7ec8e154c12",
        identity="fe7b7c093421a0e20ed40735a5accd3bbfffc893df796f54fb8138991091cfaf",
    )


def test_phara_evaluates_integer_codes_with_explicit_wrap() -> None:
    graph = AffineGraph(
        input_ids=("x0", "x1"),
        nodes=(
            AffineNode(id="twice_x0", operation="shift", inputs=("x0",), value=1),
            AffineNode(
                id="result",
                operation="subtract",
                inputs=("twice_x0", "x1"),
            ),
        ),
        output_ids=("result",),
        modulus=16,
    )

    assert evaluate(graph, input_codes=(7, 3)) == (11,)


def test_phara_direct_supertile_uses_the_pool_aligned_reference_matrix() -> None:
    analysis = analyze_direct_supertile(
        weight_codes=(
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
        ),
        aligned_bias_codes=(11, 12),
        convolution_stride=3,
        modulus=256,
    )

    assert analysis.summary == {
        "input_rows": 8,
        "convolution_rows": 2,
        "filter_lanes": 2,
        "output_values": 4,
        "constant_nodes": 4,
        "multiply_nodes": 20,
        "add_nodes": 20,
        "depth": 6,
        "max_fanout": 4,
    }
    assert analysis.proof.status == "proven"
    assert analysis.proof.reference_coefficients == (
        (1, 3, 5, 7, 9, 0, 0, 0, 11),
        (2, 4, 6, 8, 10, 0, 0, 0, 12),
        (0, 0, 0, 1, 3, 5, 7, 9, 11),
        (0, 0, 0, 2, 4, 6, 8, 10, 12),
    )
