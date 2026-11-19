import pytest

from ravel_hls import analyze, convert

from test_separable_blocks import KU5P, PACKING, conv, make_block_chain


VERIFIED = {"Verification": {"Mode": "required", "Samples": 8}}


def hidden_chain(**arguments):
    return make_block_chain([conv(3, kernel=5, stride=3), conv(4, kernel=5, stride=3)],
                            height=256, width=4, hidden=(8,), **arguments)


def test_a_hidden_dense_layer_is_recognized_as_a_head_with_a_tail():
    report = analyze(hidden_chain(), {"HLS": {}}).to_dict()

    assert report["recognition"]["head"] == "dense_0"
    assert report["recognition"]["tail"] == ["relu_2", "dense_1"]
    assert "parametrizedactivation" not in {operation["kind"] for operation in report["model_facts"]["operations"]}
    assert report["applicability"] == {"status": "applicable", "findings": []}
    stages = report["resolved_design"]["stages"]
    assert [stage["strategy"]["id"] for stage in stages[-2:]] == ["aria-dense-matrix", "hls4ml-head-tail"]
    assert stages[-2]["output"]["shape"] == [8]
    assert stages[-2]["dense"]["steps"] == (stages[-2]["input"]["words"] * stages[-2]["input"]["lanes"]
                                            // stages[-2]["dense"]["mac_lanes"]) * (8 // stages[-2]["dense"]["output_lanes"])


def test_a_hidden_dense_project_converts_bit_exactly_through_every_boundary(tmp_path):
    project = convert(hidden_chain(), tmp_path / "hidden", {"HLS": KU5P, **VERIFIED})

    manifest = project.manifest
    assert manifest["verification"]["transformation_equivalence"] == "passed"
    assert manifest["verification"]["stage_boundaries"]["status"] == "passed"
    observed = {item["tensor_id"] for item in manifest["verification"]["stage_boundaries"]["observations"]}
    assert {"dense_0:out0", "relu_2:out0"} <= observed


def test_two_products_of_a_step_share_one_multiplier_where_the_device_ports_are_known(tmp_path):
    dense = lambda design: next(stage["dense"] for stage in design["stages"] if stage["strategy"]["id"] == "aria-dense-matrix")
    known = dense(analyze(hidden_chain(), {"HLS": KU5P}).to_dict()["resolved_design"])
    unknown = dense(analyze(hidden_chain(), {"HLS": {}}).to_dict()["resolved_design"])

    assert known["packing"]["products_per_multiplier"] == 2
    assert known["products"] == known["mac_lanes"] * known["output_lanes"] == 2 * known["multipliers"]
    assert "packing" not in unknown
    assert unknown["products"] == unknown["multipliers"] == unknown["mac_lanes"] * unknown["output_lanes"]

    project = convert(hidden_chain(), tmp_path / "shared", {"HLS": KU5P, **VERIFIED})

    assert dense(project.manifest["resolved_design"])["packing"] == known["packing"]
    assert project.manifest["verification"]["transformation_equivalence"] == "passed"
    assert project.manifest["verification"]["stage_boundaries"]["status"] == "passed"
    assert "dense_matrix_shared_stream" in (project.path / "firmware/nnet_utils/ravel_dense.h").read_text()


def test_the_output_lanes_of_the_hidden_dense_follow_the_interval_target():
    relaxed = analyze(hidden_chain(), {"HLS": KU5P, "Optimization": {"TargetII": 400}}).to_dict()["resolved_design"]
    fastest = analyze(hidden_chain(), {"HLS": KU5P}).to_dict()["resolved_design"]

    dense = lambda design: next(stage["dense"] for stage in design["stages"] if stage["strategy"]["id"] == "aria-dense-matrix")
    assert dense(relaxed)["output_lanes"] <= dense(fastest)["output_lanes"]
    assert dense(relaxed)["multipliers"] <= dense(fastest)["multipliers"]


def test_a_single_block_with_a_hidden_dense_layer_converts_through_the_composed_path(tmp_path):
    model = make_block_chain([conv(3, kernel=5, stride=3)], height=64, width=4, hidden=(6,))

    project = convert(model, tmp_path / "single_hidden", {"HLS": {}, **VERIFIED})

    assert project.manifest["resolved_design"]["strategy"]["id"] == "aria-composed"
    assert project.manifest["verification"]["stage_boundaries"]["status"] == "passed"


def test_a_hidden_dense_head_works_with_two_row_packing(tmp_path):
    model = make_block_chain([conv(3), conv(4)], height=64, width=2, hidden=(5,))

    project = convert(model, tmp_path / "hidden_p2", {**PACKING, **VERIFIED})

    assert project.manifest["verification"]["transformation_equivalence"] == "passed"


def test_two_hidden_dense_layers_are_recognized_but_outside_the_support_domain():
    model = make_block_chain([conv(3), conv(4)], height=64, width=2, hidden=(6, 4))

    report = analyze(model, PACKING).to_dict()

    assert report["recognition"]["tail"] == ["relu_2", "dense_1", "relu_3", "dense_2"]
    assert report["applicability"]["status"] == "unsupported"
    assert {item["code"] for item in report["applicability"]["findings"]} == {"family.support.head_form"}
