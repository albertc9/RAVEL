import pytest

from ravel_hls import analyze, convert

from test_separable_blocks import conv, make_block_chain


VERIFIED = {"HLS": {}, "Verification": {"Mode": "required", "Samples": 8}}


def test_the_default_packing_derives_a_fill_cycle_schedule_for_a_noncanonical_first_block(
    noncanonical_geometry_model,
):
    report = analyze(noncanonical_geometry_model, {"HLS": {}}).to_dict()

    assert report["applicability"] == {"status": "applicable", "findings": []}
    scheduler = report["resolved_design"]["implementation_plan"]["phara"]["scheduler"]
    assert (scheduler["id"], scheduler["version"]) == ("row-credit", 2)
    assert scheduler["cycles"] == 31 + scheduler["fill_cycles"] + scheduler["drain_cycles"]
    assert report["resolved_design"]["rendering"]["phara_fused_function"] == "phara_pool_aligned_hybrid_p8_scheduled_cl"


def test_the_canonical_first_block_keeps_its_qualified_schedule_and_lowering():
    report = analyze(make_block_chain([conv(3, kernel=5, stride=3), conv(4, kernel=5, stride=3)],
                                      height=256, width=4), {"HLS": {}}).to_dict()

    design = report["resolved_design"]
    assert design["implementation_plan"]["phara"]["scheduler"] == {
        "id": "row-credit", "version": 1, "buffer_rows": 16, "max_live_rows": 14, "read_cycles": 32}
    assert design["rendering"]["phara_fused_function"] == "phara_pool_aligned_hybrid_p8_cl"


def test_a_single_block_with_a_noncanonical_kernel_converts_bit_exactly_at_the_default_packing(
    tmp_path, noncanonical_geometry_model,
):
    project = convert(noncanonical_geometry_model, tmp_path / "single_k3s2", VERIFIED)

    assert project.manifest["resolved_design"]["strategy"]["id"] == "phara"
    assert project.manifest["verification"]["transformation_equivalence"] == "passed"


@pytest.mark.parametrize("name, blocks", [
    ("k7s2", [conv(3, kernel=7, stride=2), conv(4)]),
    ("k5s1", [conv(3, kernel=5, stride=1), conv(4, kernel=5)]),
    ("k9s4", [conv(2, kernel=9, stride=4), conv(3, kernel=2, stride=1)]),
    ("separable", [conv(3, kernel=7, stride=2), conv(3, kernel=7, stride=2, relu=False, pool=False, diagonal=True),
                   conv(4, kernel=1, stride=1)]),
])
def test_noncanonical_first_blocks_convert_bit_exactly_through_every_boundary(tmp_path, name, blocks):
    project = convert(make_block_chain(blocks, height=128, width=4), tmp_path / name, VERIFIED)

    design = project.manifest["resolved_design"]
    assert design["stages"][0]["strategy"]["id"] == "phara"
    assert design["implementation_plan"]["phara"]["scheduler"]["version"] == 2
    assert project.manifest["verification"]["stage_boundaries"]["status"] == "passed"
