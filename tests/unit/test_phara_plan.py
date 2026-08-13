from ravel_hls.profiles.aria.plan import build_implementation_plan


def _canonical_model_facts() -> dict[str, object]:
    return {
        "operations": [
            {
                "id": "input_0",
                "outputs": [{"shape": [256, 4]}],
            },
            {
                "id": "conv2d_0",
                "attributes": {
                    "filt_height": 5,
                    "filt_width": 1,
                    "n_chan": 1,
                    "n_filt": 7,
                    "stride_height": 3,
                    "out_width": 4,
                },
            },
            {
                "id": "max_pool2d_0",
                "attributes": {"pool_height": 2},
            },
        ],
        "dense": [
            {
                "n_in": 1176,
                "n_out": 1,
                "input_group_size": 7,
                "parameter_representation": "dense",
                "feature_ordering": {"kind": "identity"},
                "numeric": {
                    "input": {"kind": "fixed"},
                    "output": {"kind": "fixed"},
                    "weight": {"kind": "fixed", "width": 7},
                    "bias": {"kind": "fixed"},
                    "accumulator": {"kind": "fixed"},
                },
            }
        ],
    }


def test_phara_p4_d4_plan_balances_the_fused_and_dense_stages() -> None:
    plan = build_implementation_plan(
        {"TemporalPacking": 4, "DenseParallelism": 4},
        _canonical_model_facts(),
    )

    assert plan["template_profile"] == "aria-phara-p4-q1-d4-v1"
    assert plan["phara"] == {
        "version": 1,
        "pool_rows_per_supertile": 2,
        "supertile_input_rows": 8,
        "pooled_words": 42,
        "stage_cycles": {"input": 64, "fused_region": 64, "dense": 42},
        "structural_ii_lower_bound": 64,
        "realization": "direct",
    }
    assert plan["first_convolution"]["multiplier_limit"] == 280
    assert plan["weight_delivery"]["mac_lanes"] == 28
    assert plan["weight_delivery"]["word_bits"] == 196
    assert plan["weight_delivery"]["depth"] == 42


def test_phara_p8_d4_plan_reaches_the_42_cycle_structural_bound() -> None:
    plan = build_implementation_plan(
        {"TemporalPacking": 8, "DenseParallelism": 4},
        _canonical_model_facts(),
    )

    assert plan["template_profile"] == "aria-phara-p8-q1-d4-v1"
    assert plan["values_per_input_word"] == 32
    assert plan["input_words_per_inference"] == 32
    assert plan["phara"] == {
        "version": 1,
        "pool_rows_per_supertile": 2,
        "supertile_input_rows": 8,
        "pooled_words": 42,
        "stage_cycles": {"input": 32, "fused_region": 42, "dense": 42},
        "structural_ii_lower_bound": 42,
        "realization": "hybrid",
        "scheduler": {
            "id": "row-credit",
            "version": 1,
            "buffer_rows": 16,
            "max_live_rows": 14,
            "read_cycles": 32,
        },
    }
