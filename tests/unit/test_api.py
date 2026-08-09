from pathlib import Path
from typing import Any

import pytest

from ravel_hls import CompatibilityError, ProjectGenerationError, optimize_project


class _FakeHlsConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        self.config = values


class _FakePrecision:
    def __init__(self, cpp: str) -> None:
        self.cpp = cpp

    def definition_cpp(self) -> str:
        return self.cpp


class _FakeType:
    def __init__(self, name: str, precision: str, n_elem: int = 1) -> None:
        self.name = name
        self.precision = _FakePrecision(precision)
        self.n_elem = n_elem
        self.n_pack = 1
        self.unpack = False

    def definition_cpp(self) -> str:
        if self.n_elem == 1:
            return f"typedef {self.precision.definition_cpp()} {self.name};\n"
        return (
            f"typedef nnet::array<{self.precision.definition_cpp()}, "
            f"{self.n_elem}*1> {self.name};\n"
        )


class _FakeVariable:
    def __init__(self, name: str, variable_type: _FakeType) -> None:
        self.name = name
        self.type = variable_type


class _FakeWeight:
    def __init__(self, name: str, type_name: str, length: int) -> None:
        self.name = name
        self.type = _FakeType(type_name, "ap_fixed<8,2>")
        self.data_length = length


class _FakeLayer:
    def __init__(self, class_name: str, **attributes: Any) -> None:
        self.class_name = class_name
        self.name = attributes.pop("name", class_name.lower())
        output_name = attributes.pop("output_name", self.name)
        type_name = attributes.pop("type_name", f"{self.name}_t")
        precision = attributes.pop("precision", "ap_fixed<16,6>")
        n_elem = attributes.pop("n_elem", 1)
        self._output_variable = _FakeVariable(
            output_name, _FakeType(type_name, precision, n_elem)
        )
        self.types = {"result_t": self._output_variable.type}
        self._weights = attributes.pop("weights", [])
        self.attributes = attributes

    def get_attr(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def get_output_variable(self) -> _FakeVariable:
        return self._output_variable

    def get_weights(self) -> list[_FakeWeight]:
        return self._weights


class _FakeHlsModel:
    def __init__(self, output_dir: Path, **overrides: Any) -> None:
        values: dict[str, Any] = {
            "Backend": "Vitis",
            "IOType": "io_stream",
            "OutputDir": str(output_dir),
            "ProjectName": "aria_top",
            "InputShapes": {"input": [256, 4]},
            "OutputShapes": {"output": [1]},
            "HLSConfig": {"Model": {"Strategy": "Latency", "ReuseFactor": 1}},
        }
        values.update(overrides)
        self.config = _FakeHlsConfig(values)
        self.write_called = False
        self.layers = [
            _FakeLayer(
                "Input",
                name="input",
                output_name="input",
                type_name="input_t",
                precision="ap_fixed<9,4>",
                n_elem=4,
                index=1,
            ),
            _FakeLayer(
                "Repack",
                name="repack_reshape",
                output_name="layer10_out",
                type_name="reshape_t",
                target_shape=[256, 4, 1],
                index=10,
            ),
            _FakeLayer(
                "Conv2D",
                name="conv",
                output_name="layer4_out",
                type_name="conv_t",
                n_elem=7,
                weights=[_FakeWeight("w4", "conv_weight_t", 35), _FakeWeight("b4", "conv_bias_t", 7)],
                index=4,
                in_height=256,
                in_width=4,
                n_chan=1,
                filt_height=5,
                filt_width=1,
                n_filt=7,
                stride_height=3,
                stride_width=1,
                pad_top=0,
                pad_bottom=0,
                pad_left=0,
                pad_right=0,
                out_height=84,
                out_width=4,
            ),
            _FakeLayer(
                "Activation",
                name="relu",
                output_name="layer5_out",
                type_name="relu_t",
                precision="ap_ufixed<15,5>",
                n_elem=7,
                activation="relu",
                n_in=2352,
                index=5,
            ),
            _FakeLayer(
                "Pooling2D",
                name="pool",
                output_name="layer6_out",
                type_name="pool_t",
                precision="ap_fixed<9,4>",
                n_elem=7,
                index=6,
                in_height=84,
                in_width=4,
                n_filt=7,
                pool_height=2,
                pool_width=1,
                stride_height=2,
                stride_width=1,
                pad_top=0,
                pad_bottom=0,
                pad_left=0,
                pad_right=0,
                pool_op="Max",
                out_height=42,
                out_width=4,
            ),
            _FakeLayer(
                "Reshape",
                name="flatten",
                output_name="layer7_out",
                type_name="pool_t",
                precision="ap_fixed<9,4>",
                n_elem=7,
                target_shape=[1176],
                index=7,
            ),
            _FakeLayer(
                "Dense",
                name="dense",
                output_name="layer9_out",
                type_name="result_t",
                precision="ap_fixed<22,11>",
                weights=[_FakeWeight("w9", "dense_weight_t", 1176), _FakeWeight("b9", "dense_bias_t", 1)],
                n_in=1176,
                n_out=1,
                index=9,
            ),
        ]

    def write(self) -> None:
        self.write_called = True
        output_dir = Path(self.config.config["OutputDir"])
        (output_dir / "firmware").mkdir(parents=True)
        (output_dir / "firmware" / "aria_top.cpp").write_text(
            "void aria_top() {}\n", encoding="utf-8"
        )
        (output_dir / "firmware" / "aria_top.h").write_text(
            "void aria_top();\n", encoding="utf-8"
        )
        (output_dir / "firmware" / "defines.h").write_text(
            "#ifndef DEFINES_H_\n#define DEFINES_H_\n#endif\n", encoding="utf-8"
        )
        (output_dir / "firmware" / "parameters.h").write_text(
            "// baseline parameters\n", encoding="utf-8"
        )
        (output_dir / "hls4ml_config.yml").write_text(
            "Backend: Vitis\nIOType: io_stream\n", encoding="utf-8"
        )

    def get_layers(self) -> list[_FakeLayer]:
        return self.layers


def test_optimize_project_rejects_unsupported_backend_before_generation(
    tmp_path: Path,
) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project", Backend="Vivado")

    with pytest.raises(CompatibilityError, match="Backend.*Vitis"):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


def test_optimize_project_rejects_parallel_io_before_generation(tmp_path: Path) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project", IOType="io_parallel")

    with pytest.raises(CompatibilityError, match="IOType.*io_stream"):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        ({"Strategy": "Resource", "ReuseFactor": 1}, "Strategy.*Latency"),
        ({"Strategy": "Latency", "ReuseFactor": 2}, "ReuseFactor.*1"),
    ],
)
def test_optimize_project_rejects_unsupported_hls_strategy(
    tmp_path: Path, model_config: dict[str, Any], message: str
) -> None:
    hls_model = _FakeHlsModel(
        tmp_path / "project", HLSConfig={"Model": model_config}
    )

    with pytest.raises(CompatibilityError, match=message):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


def test_optimize_project_rejects_incompatible_logical_input_shape(tmp_path: Path) -> None:
    hls_model = _FakeHlsModel(
        tmp_path / "project", InputShapes={"input": [128, 4]}
    )

    with pytest.raises(CompatibilityError, match=r"input shape.*\[256, 4\]"):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


def test_optimize_project_rejects_incompatible_logical_output_shape(tmp_path: Path) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project", OutputShapes={"output": [2]})

    with pytest.raises(CompatibilityError, match=r"output shape.*\[1\]"):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


def test_optimize_project_rejects_incompatible_convolution_geometry(tmp_path: Path) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project")
    hls_model.layers[2].attributes["n_filt"] = 8

    with pytest.raises(CompatibilityError, match="Conv2D.n_filt.*7"):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


def test_optimize_project_rejects_a_different_layer_sequence(tmp_path: Path) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project")
    hls_model.layers.insert(-1, _FakeLayer("Dropout", name="dropout"))

    with pytest.raises(CompatibilityError, match="layer sequence"):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


@pytest.mark.parametrize(
    ("layer_index", "attribute", "invalid_value", "message"),
    [
        (1, "target_shape", [128, 8, 1], "Repack.target_shape"),
        (2, "filt_height", 3, "Conv2D.filt_height"),
        (3, "activation", "sigmoid", "Activation.activation"),
        (4, "pool_height", 3, "Pooling2D.pool_height"),
        (5, "target_shape", [1175], "Reshape.target_shape"),
        (6, "n_in", 1175, "Dense.n_in"),
    ],
)
def test_optimize_project_rejects_noncanonical_layer_geometry(
    tmp_path: Path,
    layer_index: int,
    attribute: str,
    invalid_value: Any,
    message: str,
) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project")
    hls_model.layers[layer_index].attributes[attribute] = invalid_value

    with pytest.raises(CompatibilityError, match=message):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


def test_optimize_project_publishes_a_complete_aria_project(tmp_path: Path) -> None:
    output_dir = tmp_path / "aria_project"
    hls_model = _FakeHlsModel(output_dir)

    project = optimize_project(
        hls_model,
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )

    assert project.path == output_dir
    assert hls_model.write_called is True
    assert (output_dir / "hls4ml_config.yml").is_file()
    assert (output_dir / "ravel_config.yml").is_file()
    assert (output_dir / "ravel_manifest.json").is_file()
    optimized_source = (output_dir / "firmware" / "aria_top.cpp").read_text(
        encoding="utf-8"
    )
    assert "first_conv_2row_4lane_temporal_wide_cl" in optimized_source
    assert "maxpool2d_wide_nonoverlap_cl" in optimized_source
    assert "dense_wide_stream" in optimized_source
    assert "nnet::repack_stream" not in optimized_source
    bridge = (output_dir / "aria_top_bridge.cpp").read_text(encoding="utf-8")
    assert "hls::stream<input_x2_t>" in bridge
    assert "for (unsigned pair = 0; pair < 128; pair++)" in bridge
    testbench = (output_dir / "aria_top_test.cpp").read_text(encoding="utf-8")
    assert "pack_aria_test_input" in testbench
    assert "hls::stream<input_x2_t>" in testbench
    assert project.implementation_plan["temporal_pack"] == 2
    assert project.implementation_plan["width_lanes"] == 4
    assert [item["id"] for item in project.manifest["pipeline"]["passes"]] == [
        "PackTemporalInput2x",
        "FuseRepackReshapeIntoFirstConv",
        "PropagateWideReLUStream",
        "SpecializeNonOverlappingMaxPool",
        "StreamFlattenIntoDense",
        "BindShallowInternalFifos",
    ]
    assert project.status == {
        "generation": "complete",
        "dependency_qualification": "qualified",
        "correctness_verification": "not_run",
        "model_fidelity": "not_run",
        "source_integrity": "clean",
        "performance_qualification": "not_run",
    }


def test_optimize_project_cleanly_replaces_a_recognized_ravel_project(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "aria_project"
    optimize_project(
        _FakeHlsModel(output_dir),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    managed_header = output_dir / "firmware" / "nnet_utils" / "nnet_aria.h"
    managed_header.write_text("manual edit\n", encoding="utf-8")

    project = optimize_project(
        _FakeHlsModel(output_dir),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )

    assert "first_conv_2row_4lane_temporal_wide_cl" in managed_header.read_text(
        encoding="utf-8"
    )
    assert project.status["source_integrity"] == "clean"


def test_optimize_project_requires_force_to_replace_an_unrecognized_target(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "aria_project"
    output_dir.mkdir()
    (output_dir / "unmanaged.txt").write_text("unmanaged\n", encoding="utf-8")

    with pytest.raises(ProjectGenerationError, match="unrecognized target"):
        optimize_project(
            _FakeHlsModel(output_dir),
            config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
        )

    project = optimize_project(
        _FakeHlsModel(output_dir),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
        force_replace=True,
    )

    assert project.path == output_dir
    assert not (output_dir / "unmanaged.txt").exists()
