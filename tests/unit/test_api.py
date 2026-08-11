import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ravel_hls import (
    BuildError,
    CompatibilityError,
    ConfigurationError,
    Project,
    ProjectGenerationError,
    VerificationError,
    convert,
    Parameters,
)
from ravel_hls.api import convert_from_keras_model, optimize_project, refresh_model


class _FakeHlsConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        self.config = values


class _FakePrecision:
    def __init__(self, cpp: str) -> None:
        self.cpp = cpp
        arguments = cpp.split("<", 1)[1].rsplit(">", 1)[0].split(",")
        self.width = int(arguments[0])
        self.integer = int(arguments[1])
        self.fractional = self.width - self.integer
        self.signed = not cpp.startswith("ap_ufixed")
        self.rounding_mode = arguments[2] if len(arguments) > 2 else "TRN"
        self.saturation_mode = arguments[3] if len(arguments) > 3 else "WRAP"
        self.saturation_bits = int(arguments[4]) if len(arguments) > 4 else 0

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
        self.data = np.zeros(length, dtype=np.float32)


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
        accum_precision = attributes.pop("accum_precision", None)
        if accum_precision is not None:
            self.attributes = {
                "accum_t": _FakeType(f"{self.name}_accum_t", accum_precision)
            }
        else:
            self.attributes = {}
        self._weights = attributes.pop("weights", [])
        self.attributes.update(attributes)
        self._input_variable: _FakeVariable | None = None

    def get_attr(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def get_output_variable(self) -> _FakeVariable:
        return self._output_variable

    def get_input_variable(self) -> _FakeVariable:
        assert self._input_variable is not None
        return self._input_variable

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
                module="hgq.layers.conv",
                reuse_factor=1,
                strategy="latency",
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
                accum_precision="ap_fixed<22,11>",
                module="hgq.layers.core.dense",
                reuse_factor=1,
                strategy="latency",
                weights=[_FakeWeight("w9", "dense_weight_t", 1176), _FakeWeight("b9", "dense_bias_t", 1)],
                n_in=1176,
                n_out=1,
                index=9,
            ),
        ]
        for previous, current in zip(self.layers, self.layers[1:]):
            current._input_variable = previous.get_output_variable()

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
            "Backend: Vitis\n"
            "IOType: io_stream\n"
            "ProjectName: aria_top\n"
            f"OutputDir: {output_dir}\n"
            f"KerasModel: !keras_model '{output_dir}/keras_model.keras'\n"
            "HLSConfig:\n  Model:\n    Strategy: Latency\n    ReuseFactor: 1\n"
            "InputShapes:\n  input: [256, 4]\n"
            "OutputShapes:\n  layer9_out: [1]\n",
            encoding="utf-8",
        )
        (output_dir / "keras_model.keras").write_bytes(b"fake keras model")
        (output_dir / "build_prj.tcl").write_text(
            "source build_opt.tcl\n"
            "catch {config_array_partition -maximum_size $maximum_size}\n"
            "csynth_design\n",
            encoding="utf-8",
        )

    def get_layers(self) -> list[_FakeLayer]:
        return self.layers


class _VerifyingFakeHlsModel(_FakeHlsModel):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.compile_called = False

    def compile(self) -> None:
        self.compile_called = True

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.sum(inputs, axis=(1, 2), dtype=np.float32).reshape(-1, 1)


class _AssignableVariable:
    def __init__(self, path: str, values: list[float]) -> None:
        self.path = path
        self.name = path.rsplit("/", 1)[-1]
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self.values.copy()

    def assign(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float32)


class _ParameterLayer:
    def __init__(self, values: list[float], round_mode: str) -> None:
        self.name = "private_dense"
        self._class_name = "QDense"
        self.round_mode = round_mode
        self.weights = [
            _AssignableVariable("private_dense/kernel", values),
            _AssignableVariable("private_dense/bias", [0.0]),
            _AssignableVariable("private_dense/private_dense_kq/k", [1.0]),
            _AssignableVariable("private_dense/private_dense_kq/i", [2.0]),
            _AssignableVariable("private_dense/private_dense_kq/f", [3.0]),
        ]

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kq_conf": {
                "q_type": "kif",
                "round_mode": self.round_mode,
                "overflow_mode": "SAT_SYM",
                "homogeneous_axis": [0],
                "heterogeneous_axis": None,
            },
        }


class _ParameterModel:
    def __init__(self, values: list[float], round_mode: str = "RND") -> None:
        self.layers = [_ParameterLayer(values, round_mode)]


def test_convert_accepts_one_public_configuration_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "aria_project"
    converted_model = _FakeHlsModel(output_dir)
    conversion_call: dict[str, Any] = {}

    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        conversion_call.update(kwargs)
        return converted_model

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)
    model = object()

    project = convert(
        model,
        {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {
                "Backend": "Vitis",
                "IOType": "io_stream",
                "Part": "xcku5p-ffvb676-2-e",
                "ClockPeriod": 5,
                "Config": {"Model": {"Strategy": "Latency", "ReuseFactor": 1}},
            },
            "Verification": {"Mode": "disabled"},
        },
    )

    assert project.path == output_dir
    assert isinstance(project, Project)
    assert Project.open(output_dir).path == output_dir
    assert conversion_call == {
        "model": model,
        "output_dir": str(output_dir),
        "project_name": "aria_top",
        "hls_config": {"Model": {"Strategy": "Latency", "ReuseFactor": 1}},
        "backend": "Vitis",
        "io_type": "io_stream",
        "part": "xcku5p-ffvb676-2-e",
        "clock_period": 5,
    }
    assert project.config["Vitis"] == {
        "Run": False,
        "Stages": {
            "Reset": True,
            "CSim": False,
            "Synth": True,
            "CoSim": False,
            "Validation": False,
            "Export": False,
            "VSynth": False,
        },
    }


def test_convert_defaults_to_the_aggressive_specialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "aria_project"
    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(
        convert_from_keras_model=lambda **kwargs: _FakeHlsModel(output_dir)
    )
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    project = convert(
        object(),
        {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Verification": {"Mode": "disabled"},
        },
    )

    assert project.config["Optimization"] == {
        "TemporalPacking": 4,
        "DenseParallelism": 2,
    }


def test_convert_records_dense_shape_and_coefficient_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "aria_project"
    converted_model = _FakeHlsModel(output_dir)
    dense_kernel = converted_model.layers[-1].get_weights()[0]
    dense_kernel.data[:3] = [0.5, -0.25, 0.375]
    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(
        convert_from_keras_model=lambda **kwargs: converted_model
    )
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    project = convert(
        object(),
        {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Verification": {"Mode": "disabled"},
        },
    )

    assert project.manifest["source_model"]["facts"]["dense"] == [
        {
            "role": "output",
            "n_in": 1176,
            "n_out": 1,
            "kernel": {
                "shape": [1176],
                "elements": 1176,
                "statistics": {
                    "zero": 1173,
                    "nonzero": 3,
                    "power_of_two": 2,
                    "unique": 4,
                },
            },
            "bias": {"shape": [1], "elements": 1},
            "numeric": {
                "input": {
                    "kind": "fixed",
                    "width": 9,
                    "integer": 4,
                    "fractional": 5,
                    "signed": True,
                    "rounding": "TRN",
                    "overflow": "WRAP",
                    "saturation_bits": 0,
                },
                "output": {
                    "kind": "fixed",
                    "width": 22,
                    "integer": 11,
                    "fractional": 11,
                    "signed": True,
                    "rounding": "TRN",
                    "overflow": "WRAP",
                    "saturation_bits": 0,
                },
                "weight": {
                    "kind": "fixed",
                    "width": 8,
                    "integer": 2,
                    "fractional": 6,
                    "signed": True,
                    "rounding": "TRN",
                    "overflow": "WRAP",
                    "saturation_bits": 0,
                },
                "bias": {
                    "kind": "fixed",
                    "width": 8,
                    "integer": 2,
                    "fractional": 6,
                    "signed": True,
                    "rounding": "TRN",
                    "overflow": "WRAP",
                    "saturation_bits": 0,
                },
                "accumulator": {
                    "kind": "fixed",
                    "width": 22,
                    "integer": 11,
                    "fractional": 11,
                    "signed": True,
                    "rounding": "TRN",
                    "overflow": "WRAP",
                    "saturation_bits": 0,
                },
            },
        }
    ]


def test_convert_preserves_an_explicit_compatibility_specialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "aria_project"
    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(
        convert_from_keras_model=lambda **kwargs: _FakeHlsModel(output_dir)
    )
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    project = convert(
        object(),
        {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
            "Verification": {"Mode": "disabled"},
        },
    )

    assert project.config["Optimization"] == {
        "TemporalPacking": 2,
        "DenseParallelism": 1,
    }


@pytest.mark.parametrize(
    ("optimization", "expected"),
    [
        (
            {"TemporalPacking": 2},
            {"TemporalPacking": 2, "DenseParallelism": 2},
        ),
        (
            {"DenseParallelism": 1},
            {"TemporalPacking": 4, "DenseParallelism": 1},
        ),
    ],
)
def test_convert_defaults_only_the_omitted_specialization_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimization: dict[str, int],
    expected: dict[str, int],
) -> None:
    output_dir = tmp_path / "aria_project"
    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(
        convert_from_keras_model=lambda **kwargs: _FakeHlsModel(output_dir)
    )
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    project = convert(
        object(),
        {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Optimization": optimization,
            "Verification": {"Mode": "disabled"},
        },
    )

    assert project.config["Optimization"] == expected


def test_convert_renders_the_selected_dense_schedule_and_control_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        return _FakeHlsModel(Path(kwargs["output_dir"]))

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    def generate(output_dir: Path, dense_parallelism: int) -> Project:
        return convert(
            object(),
            {
                "Project": {"Name": "aria_top", "OutputDir": output_dir},
                "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
                "Optimization": {
                    "TemporalPacking": 2,
                    "DenseParallelism": dense_parallelism,
                },
                "Verification": {"Mode": "disabled"},
            },
        )

    dense1 = generate(tmp_path / "dense1", 1)
    dense2 = generate(tmp_path / "dense2", 2)

    dense1_header = (
        dense1.path / "firmware" / "nnet_utils" / "nnet_aria.h"
    ).read_text(encoding="utf-8")
    dense2_header = (
        dense2.path / "firmware" / "nnet_utils" / "nnet_aria.h"
    ).read_text(encoding="utf-8")
    assert "constexpr unsigned DENSE_PARALLELISM = 1;" in dense1_header
    assert "constexpr unsigned DENSE_PARALLELISM = 2;" in dense2_header
    assert "parallel_group < DENSE_PARALLELISM" in dense2_header
    for project in (dense1, dense2):
        source = (project.path / "firmware" / "aria_top.cpp").read_text(
            encoding="utf-8"
        )
        assert "#pragma HLS DATAFLOW disable_start_propagation" in source


def test_convert_renders_the_selected_temporal_packing_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        return _FakeHlsModel(Path(kwargs["output_dir"]))

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    def generate(output_dir: Path, temporal_packing: int | None) -> Project:
        config: dict[str, Any] = {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Verification": {"Mode": "disabled"},
        }
        if temporal_packing is not None:
            config["Optimization"] = {
                "TemporalPacking": temporal_packing,
                "DenseParallelism": 1,
            }
        return convert(object(), config)

    packed4 = generate(tmp_path / "packed4", None)
    packed2 = generate(tmp_path / "packed2", 2)

    packed4_source = (packed4.path / "firmware" / "aria_top.cpp").read_text(
        encoding="utf-8"
    )
    packed2_source = (packed2.path / "firmware" / "aria_top.cpp").read_text(
        encoding="utf-8"
    )
    assert "first_conv_4row_4lane_temporal_wide_cl" in packed4_source
    assert "first_conv_2row_4lane_temporal_wide_cl" in packed2_source
    packed4_bridge = (packed4.path / "aria_top_bridge.cpp").read_text(
        encoding="utf-8"
    )
    assert "hls::stream<input_x4_t>" in packed4_bridge
    assert "word_index < 64" in packed4_bridge
    assert "row < 4" in packed4_bridge
    assert packed4.implementation_plan["template_profile"] == "aria-p4-d2-v1"
    assert packed4.implementation_plan["input_words_per_inference"] == 64
    assert packed4.implementation_plan["dense_steps"] == 84
    assert packed2.implementation_plan["template_profile"] == "aria-p2-d1-v1"
    assert packed2.implementation_plan["input_words_per_inference"] == 128
    assert packed2.implementation_plan["dense_steps"] == 168
    assert packed4.manifest["interfaces"]["hls_stream_interface"] == {
        "input_rows_per_word": 4,
        "channels_per_row": 4,
        "values_per_input_word": 16,
        "input_words_per_inference": 64,
        "output_words_per_inference": 1,
        "input_scalar_bits": 9,
        "output_scalar_bits": 22,
        "ordering": "row-major; time before channel",
        "protocol": "axis",
        "block_control": "ap_ctrl_hs",
        "optional_axis_sidebands": [],
    }
    assert packed4.manifest["interfaces"]["rtl_interface"]["expected"][
        "input_tdata_bits"
    ] == 256


def test_convert_runs_vitis_when_the_configuration_enables_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "aria_project"
    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(
        convert_from_keras_model=lambda **kwargs: _FakeHlsModel(output_dir)
    )
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)
    built: list[Path] = []
    monkeypatch.setattr(Project, "build", lambda self: built.append(self.path))

    project = convert(
        object(),
        {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Verification": {"Mode": "disabled"},
            "Vitis": {"Run": True},
        },
    )

    assert project.path == output_dir
    assert built == [output_dir]


def test_convert_forwards_inputs_and_force_replacement_as_invocation_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = np.zeros((2, 256, 4), dtype=np.float32)
    converted = SimpleNamespace(path=tmp_path / "aria_project")
    invocation: dict[str, Any] = {}

    def fake_convert(model: Any, **kwargs: Any) -> Any:
        invocation.update({"model": model, **kwargs})
        return converted

    monkeypatch.setattr("ravel_hls.api.convert_from_keras_model", fake_convert)
    model = object()

    result = convert(
        model,
        {
            "Project": {
                "Name": "aria_top",
                "OutputDir": tmp_path / "aria_project",
                "ForceReplace": True,
            },
            "HLS": {"Config": {}},
            "Verification": {"Mode": "required"},
        },
        inputs=inputs,
    )

    assert result is converted
    assert invocation["model"] is model
    assert invocation["verification_inputs"] is inputs
    assert invocation["force_replace"] is True


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("Project", {"Unknown": True}, "Project.Unknown"),
        ("Project", {"ForceReplace": "yes"}, "Project.ForceReplace"),
        ("HLS", {"Unknown": True}, "HLS.Unknown"),
        ("Verification", {"Unknown": True}, "Verification.Unknown"),
        ("Verification", {"Samples": 0}, "Verification.Samples"),
    ],
)
def test_convert_validates_all_public_configuration_sections(
    tmp_path: Path, section: str, values: dict[str, Any], message: str
) -> None:
    config: dict[str, Any] = {
        "Project": {"Name": "aria_top", "OutputDir": tmp_path / "project"},
        "HLS": {"Config": {}},
    }
    config.setdefault(section, {}).update(values)

    with pytest.raises(ConfigurationError, match=message):
        convert(object(), config)


def test_convert_rejects_an_unknown_top_level_configuration_field() -> None:
    with pytest.raises(ConfigurationError, match="UnknownField"):
        convert(object(), {"UnknownField": True})


def test_convert_rejects_a_non_boolean_vitis_run_value(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Vitis.Run"):
        convert(
            object(),
            {
                "Project": {"Name": "aria_top", "OutputDir": tmp_path / "project"},
                "HLS": {"Config": {}},
                "Vitis": {"Run": "yes"},
            },
        )


def test_convert_rejects_an_unknown_vitis_stage(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Vitis.Stages.Unknown"):
        convert(
            object(),
            {
                "Project": {"Name": "aria_top", "OutputDir": tmp_path / "project"},
                "HLS": {"Config": {}},
                "Vitis": {"Stages": {"Unknown": True}},
            },
        )


def test_convert_rejects_a_non_boolean_vitis_stage(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Vitis.Stages.Synth"):
        convert(
            object(),
            {
                "Project": {"Name": "aria_top", "OutputDir": tmp_path / "project"},
                "HLS": {"Config": {}},
                "Vitis": {"Stages": {"Synth": 1}},
            },
        )


def test_optimize_project_rejects_unsupported_backend_before_generation(
    tmp_path: Path,
) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project", Backend="Vivado")

    with pytest.raises(CompatibilityError, match="Backend.*Vitis"):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


def test_optimize_project_rejects_an_unqualified_dependency_stack_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project")
    monkeypatch.setattr(
        "ravel_hls.api.inspect_dependencies",
        lambda: {
            "dependency_qualification": "failed",
            "dependencies": {
                "HGQ": {
                    "installed": "0.2.1",
                    "required": "not installed",
                    "status": "conflict",
                }
            },
        },
    )

    with pytest.raises(CompatibilityError, match="dependency stack.*HGQ"):
        optimize_project(
            hls_model,
            config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
        )

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


def test_optimize_project_requires_hgq_quantized_conv_and_dense_layers(
    tmp_path: Path,
) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project")
    hls_model.layers[2].attributes["module"] = "keras.src.layers.convolutional.conv2d"

    with pytest.raises(CompatibilityError, match="Conv2D.*HGQ"):
        optimize_project(hls_model, config={"Profile": "aria"})

    assert hls_model.write_called is False


def test_optimize_project_rejects_a_per_layer_reuse_override(tmp_path: Path) -> None:
    hls_model = _FakeHlsModel(tmp_path / "project")
    hls_model.layers[-1].attributes["reuse_factor"] = 2

    with pytest.raises(CompatibilityError, match="Dense.reuse_factor.*1"):
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
        config={
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Verification": {"Mode": "disabled"},
        },
    )

    assert project.path == output_dir
    assert hls_model.write_called is True
    assert (output_dir / "hls4ml_config.yml").is_file()
    assert (output_dir / "ravel_config.yml").is_file()
    assert (output_dir / "ravel_manifest.json").is_file()
    assert (output_dir / "build_opt.tcl").read_text(encoding="utf-8") == (
        "array set opt {\n"
        "    reset      1\n"
        "    csim       0\n"
        "    synth      1\n"
        "    cosim      0\n"
        "    validation 0\n"
        "    export     0\n"
        "    vsynth     0\n"
        "    fifo_opt   0\n"
        "}\n"
    )
    published_build_script = (output_dir / "build_prj.tcl").read_text(
        encoding="utf-8"
    )
    assert "config_array_partition -maximum_size" not in published_build_script
    assert "csynth_design" in published_build_script
    assert project.manifest["schema_version"] == 2
    assert project.manifest["ravel"]["release"] == "1.3.0"
    published_hls_config = (output_dir / "hls4ml_config.yml").read_text(encoding="utf-8")
    assert "OutputDir: .\n" in published_hls_config
    assert str(output_dir) not in published_hls_config
    assert "KerasModel: !keras_model 'keras_model.keras'" in published_hls_config
    assert ".ravel-" not in published_hls_config
    published_ravel_config = (output_dir / "ravel_config.yml").read_text(
        encoding="utf-8"
    )
    assert "OutputDir: .\n" in published_ravel_config
    assert str(output_dir) not in published_ravel_config
    assert str(output_dir) not in json.dumps(project.manifest, sort_keys=True)
    optimized_source = (output_dir / "firmware" / "aria_top.cpp").read_text(
        encoding="utf-8"
    )
    assert "first_conv_4row_4lane_temporal_wide_cl" in optimized_source
    assert "maxpool2d_wide_nonoverlap_cl" in optimized_source
    assert "dense_wide_stream" in optimized_source
    assert "nnet::repack_stream" not in optimized_source
    bridge = (output_dir / "aria_top_bridge.cpp").read_text(encoding="utf-8")
    assert "hls::stream<input_x4_t>" in bridge
    assert "word_index < 64" in bridge
    assert "PRAGMA_DATA_PACK" not in bridge
    testbench = (output_dir / "aria_top_test.cpp").read_text(encoding="utf-8")
    assert "pack_aria_test_input" in testbench
    assert "hls::stream<input_x4_t>" in testbench
    assert "PRAGMA_DATA_PACK" not in testbench
    assert project.implementation_plan["temporal_pack"] == 4
    assert project.implementation_plan["dense_parallelism"] == 2
    assert project.implementation_plan["width_lanes"] == 4
    assert project.manifest["interfaces"]["rtl_interface"] == {
        "expected": {
            "qualification_profile": "hls4ml-1.2.0-vitis-2023.2-axis-packing-v1",
            "input_tdata_bits": 256,
            "output_tdata_bits": 32,
            "input_tdata_port": "input_TDATA",
            "output_tdata_port": "layer9_out_TDATA",
            "input_scalar_bits": 9,
            "output_scalar_bits": 22,
        },
        "measured": None,
    }
    source_closure = {
        entry["path"]: entry for entry in project.manifest["source_closure"]
    }
    assert {
        "keras_model.keras",
        "hls4ml_config.yml",
        "ravel_config.yml",
        "firmware/parameters.h",
        "firmware/aria_top.cpp",
        "firmware/nnet_utils/nnet_aria.h",
    } <= set(source_closure)
    assert source_closure["keras_model.keras"] == {
        "role": "model",
        "path": "keras_model.keras",
        "size": 16,
        "sha256": hashlib.sha256(b"fake keras model").hexdigest(),
    }
    assert len(project.manifest["source_closure_sha256"]) == 64
    assert [item["id"] for item in project.manifest["pipeline"]["passes"]] == [
        "PackTemporalInput4x",
        "FuseRepackReshapeIntoFirstConv",
        "PropagateWideReLUStream",
        "SpecializeNonOverlappingMaxPool",
        "StreamFlattenIntoDense",
        "BindShallowInternalFifos",
        "ElideDataflowStartPropagation",
    ]
    assert all(
        {
            "id",
            "version",
            "order",
            "legality",
            "resolved_parameters",
            "input_ir_sha256",
            "output_ir_sha256",
            "affected_artifacts",
        }
        <= item.keys()
        for item in project.manifest["pipeline"]["passes"]
    )
    assert project.status == {
        "generation": "complete",
        "dependency_qualification": "qualified",
        "correctness_verification": "not_run",
        "model_fidelity": "not_run",
        "source_integrity": "clean",
        "performance_qualification": "not_run",
    }


def test_project_build_runs_vitis_in_place_and_records_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = optimize_project(
        _FakeHlsModel(tmp_path / "aria_project"),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    invocation: dict[str, Any] = {}
    record = object()

    monkeypatch.setattr(
        shutil,
        "which",
        lambda command: "/opt/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
    )

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        invocation.update({"args": args, **kwargs})
        return subprocess.CompletedProcess(args, 0, "synthesis complete\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(Project, "record", lambda self, report_dir: record)

    result = project.build()

    assert result is record
    assert invocation == {
        "args": [
            "/opt/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
            "-f",
            "build_prj.tcl",
        ],
        "cwd": project.path,
        "check": False,
        "capture_output": True,
        "text": True,
        "shell": False,
    }
    assert (project.path / "ravel_vitis.log").read_text(encoding="utf-8") == (
        "synthesis complete\n"
    )


def test_project_build_failure_keeps_the_log_without_recording_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = optimize_project(
        _FakeHlsModel(tmp_path / "aria_project"),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    monkeypatch.setattr(shutil, "which", lambda command: "/tools/vitis_hls")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1, "starting\n", "synthesis failed\n"
        ),
    )
    recorded: list[Path] = []
    monkeypatch.setattr(Project, "record", lambda self, path: recorded.append(path))

    with pytest.raises(BuildError, match="exit code 1"):
        project.build()

    assert (project.path / "ravel_vitis.log").read_text(encoding="utf-8") == (
        "starting\nsynthesis failed\n"
    )
    assert recorded == []
    assert not (project.path / "ravel_qualification.json").exists()


def test_project_build_reports_a_missing_vitis_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = optimize_project(
        _FakeHlsModel(tmp_path / "aria_project"),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    monkeypatch.setattr(shutil, "which", lambda command: None)

    with pytest.raises(BuildError, match="vitis_hls"):
        project.build()

    assert not (project.path / "ravel_vitis.log").exists()
    assert not (project.path / "ravel_qualification.json").exists()


def test_project_build_rejects_modified_sources_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = optimize_project(
        _FakeHlsModel(tmp_path / "aria_project"),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    (project.path / "firmware" / "aria_top.cpp").write_text(
        "void modified() {}\n", encoding="utf-8"
    )
    launched: list[str] = []
    monkeypatch.setattr(shutil, "which", lambda command: launched.append(command))

    with pytest.raises(VerificationError, match="modified RAVEL project"):
        project.build()

    assert launched == []


def test_generation_writes_the_selected_vitis_stages(tmp_path: Path) -> None:
    output_dir = tmp_path / "aria_project"

    optimize_project(
        _FakeHlsModel(output_dir),
        config={
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {}},
            "Verification": {"Mode": "disabled"},
            "Vitis": {
                "Run": False,
                "Stages": {"Reset": False, "CSim": True, "CoSim": True},
            },
        },
    )

    options = (output_dir / "build_opt.tcl").read_text(encoding="utf-8")
    assert "    reset      0\n" in options
    assert "    csim       1\n" in options
    assert "    synth      1\n" in options
    assert "    cosim      1\n" in options
    assert "    export     0\n" in options
    assert "    vsynth     0\n" in options


def test_source_integrity_prunes_the_vendor_build_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = optimize_project(
        _FakeHlsModel(tmp_path / "aria_project"),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    vendor_file = project.path / "aria_top_prj" / "solution1" / "syn" / "report.xml"
    vendor_file.parent.mkdir(parents=True)
    vendor_file.write_text("large vendor artifact\n", encoding="utf-8")
    original_stat = Path.stat

    def guarded_stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        if any(part.endswith("_prj") for part in path.parts):
            raise AssertionError("source integrity entered the vendor build tree")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)

    assert project.status["source_integrity"] == "clean"


def test_generation_identity_changes_when_model_parameters_change(
    tmp_path: Path,
) -> None:
    first_model = _FakeHlsModel(tmp_path / "first")
    second_model = _FakeHlsModel(tmp_path / "second")
    second_model.layers[2].get_weights()[0].data[0] = 1.0

    first = optimize_project(
        first_model,
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    second = optimize_project(
        second_model,
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )

    assert (
        first.manifest["source_model"]["semantic_model_sha256"]
        != second.manifest["source_model"]["semantic_model_sha256"]
    )
    assert (
        first.manifest["generation_fingerprint"]
        != second.manifest["generation_fingerprint"]
    )


def test_published_project_can_move_without_revealing_its_generation_path(
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "private-user" / "aria_project"
    project = optimize_project(
        _FakeHlsModel(original_path),
        config={
            "Project": {"Name": "aria_top", "OutputDir": original_path},
            "HLS": {"Config": {}},
            "Verification": {"Mode": "disabled"},
        },
    )
    moved_path = tmp_path / "recipient" / "portable_project"
    shutil.copytree(project.path, moved_path)

    moved = Project.open(moved_path)

    assert moved.config["Project"]["OutputDir"] == "."
    assert moved.status["source_integrity"] == "clean"
    assert callable(moved.link().compile)


def test_generation_identity_excludes_the_output_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        return _FakeHlsModel(Path(kwargs["output_dir"]))

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    def config(output_dir: Path) -> dict[str, Any]:
        return {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency", "ReuseFactor": 1}}},
            "Verification": {"Mode": "disabled"},
        }

    first = convert(object(), config(tmp_path / "first"))
    second = convert(object(), config(tmp_path / "second"))

    assert first.manifest["configuration_sha256"] == second.manifest[
        "configuration_sha256"
    ]
    assert first.manifest["generation_fingerprint"] == second.manifest[
        "generation_fingerprint"
    ]


def test_generation_identity_records_the_resolved_specialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        return _FakeHlsModel(Path(kwargs["output_dir"]))

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    def config(
        output_dir: Path, optimization: dict[str, int] | None = None
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Verification": {"Mode": "disabled"},
        }
        if optimization is not None:
            values["Optimization"] = optimization
        return values

    aggressive = convert(object(), config(tmp_path / "aggressive"))
    compatibility = convert(
        object(),
        config(
            tmp_path / "compatibility",
            {"TemporalPacking": 2, "DenseParallelism": 1},
        ),
    )

    assert aggressive.manifest["generation_configuration"]["ravel"] == {
        "Profile": "aria",
        "OptimizationPolicy": "aria-aggressive-v1",
        "Optimization": {"TemporalPacking": 4, "DenseParallelism": 2},
    }
    assert compatibility.manifest["generation_configuration"]["ravel"] == {
        "Profile": "aria",
        "OptimizationPolicy": "aria-aggressive-v1",
        "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
    }
    assert aggressive.manifest["configuration_sha256"] != compatibility.manifest[
        "configuration_sha256"
    ]
    assert aggressive.manifest["generation_fingerprint"] != compatibility.manifest[
        "generation_fingerprint"
    ]


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


def test_project_links_a_restricted_hls4ml_existing_project(
    tmp_path: Path,
) -> None:
    project = optimize_project(
        _FakeHlsModel(tmp_path / "aria_project"),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )

    linked = project.link()

    assert callable(linked.compile)
    assert callable(linked.predict)
    assert callable(linked.build)
    with pytest.raises(Exception, match='method "write"'):
        linked.write()


def test_convert_from_keras_model_delegates_to_the_aria_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "aria_project"
    converted_model = _FakeHlsModel(output_dir)
    conversion_call: dict[str, Any] = {}

    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        conversion_call.update(kwargs)
        return converted_model

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)
    keras_model = object()

    project = convert_from_keras_model(
        keras_model,
        output_dir=output_dir,
        project_name="aria_top",
        hls_config={"Model": {"Strategy": "Latency", "ReuseFactor": 1}},
        ravel_config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
        backend="Vitis",
        io_type="io_stream",
        part="xcku5p-ffvb676-2-e",
        clock_period=5,
    )

    assert project.path == output_dir
    assert conversion_call == {
        "model": keras_model,
        "output_dir": str(output_dir),
        "project_name": "aria_top",
        "hls_config": {"Model": {"Strategy": "Latency", "ReuseFactor": 1}},
        "backend": "Vitis",
        "io_type": "io_stream",
        "part": "xcku5p-ffvb676-2-e",
        "clock_period": 5,
    }


def test_convert_from_keras_model_loads_a_keras_path_with_hgq2_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "aria_project"
    model_path = tmp_path / "trained.keras"
    model_path.write_bytes(b"model")
    loaded_model = object()
    load_call: dict[str, Any] = {}
    conversion_call: dict[str, Any] = {}

    def fake_load(path: Path, **kwargs: Any) -> object:
        load_call.update({"path": path, **kwargs})
        return loaded_model

    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        conversion_call.update(kwargs)
        return _FakeHlsModel(output_dir)

    qconv2d = object()
    qdense = object()
    fake_keras = ModuleType("keras")
    fake_keras.models = SimpleNamespace(load_model=fake_load)
    fake_hgq = ModuleType("hgq")
    fake_hgq_layers = ModuleType("hgq.layers")
    fake_hgq_layers.QConv2D = qconv2d
    fake_hgq_layers.QDense = qdense
    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "keras", fake_keras)
    monkeypatch.setitem(sys.modules, "hgq", fake_hgq)
    monkeypatch.setitem(sys.modules, "hgq.layers", fake_hgq_layers)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    convert_from_keras_model(
        model_path,
        output_dir=output_dir,
        project_name="aria_top",
        hls_config={"Model": {"Strategy": "Latency", "ReuseFactor": 1}},
        ravel_config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )

    assert load_call == {
        "path": model_path,
        "custom_objects": {"QConv2D": qconv2d, "QDense": qdense},
    }
    assert conversion_call["model"] is loaded_model


def test_disabled_verification_rejects_supplied_inputs(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="verification_inputs"):
        optimize_project(
            _FakeHlsModel(tmp_path / "aria_project"),
            config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
            verification_inputs=object(),
        )


def test_required_verification_records_bit_exact_transformation_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hls4ml.utils.link

    class MatchingExistingProject:
        def __init__(self, project_path: Path) -> None:
            self.project_path = project_path

        def compile(self) -> None:
            pass

        def predict(self, inputs: np.ndarray) -> np.ndarray:
            return np.sum(inputs, axis=(1, 2), dtype=np.float32).reshape(-1, 1)

    monkeypatch.setattr(
        hls4ml.utils.link, "FilesystemModelGraph", MatchingExistingProject
    )
    hls_model = _VerifyingFakeHlsModel(tmp_path / "aria_project")

    project = optimize_project(
        hls_model,
        config={
            "Profile": "aria",
            "Verification": {"Mode": "required", "Samples": 2, "Seed": 7},
        },
    )

    assert hls_model.compile_called is True
    assert project.status["correctness_verification"] == "passed"
    assert project.manifest["verification"]["transformation_equivalence"] == "passed"
    expected_inputs = np.random.default_rng(7).uniform(
        -1.0, 1.0, size=(2, 256, 4)
    ).astype(np.float32)
    assert project.manifest["verification"]["stimuli"] == {
        "kind": "synthetic",
        "shape": [2, 256, 4],
        "dtype": "float32",
        "sample_count": 2,
        "seed": 7,
        "content_sha256": hashlib.sha256(expected_inputs.tobytes()).hexdigest(),
    }


def test_transformation_mismatch_prevents_project_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hls4ml.utils.link

    class MismatchingExistingProject:
        def __init__(self, project_path: Path) -> None:
            self.project_path = project_path

        def compile(self) -> None:
            pass

        def predict(self, inputs: np.ndarray) -> np.ndarray:
            baseline = np.sum(inputs, axis=(1, 2), dtype=np.float32).reshape(-1, 1)
            return baseline + 1.0

    monkeypatch.setattr(
        hls4ml.utils.link, "FilesystemModelGraph", MismatchingExistingProject
    )
    output_dir = tmp_path / "aria_project"

    with pytest.raises(VerificationError, match="equivalence failed"):
        optimize_project(
            _VerifyingFakeHlsModel(output_dir),
            config={"Profile": "aria", "Verification": {"Mode": "required"}},
        )

    assert not output_dir.exists()


def test_required_compile_failure_is_a_verification_error(tmp_path: Path) -> None:
    class FailingCompileModel(_VerifyingFakeHlsModel):
        def compile(self) -> None:
            raise RuntimeError("compiler failed")

    output_dir = tmp_path / "aria_project"

    with pytest.raises(VerificationError, match="baseline compilation"):
        optimize_project(
            FailingCompileModel(output_dir),
            config={"Profile": "aria", "Verification": {"Mode": "required"}},
        )

    assert not output_dir.exists()


def test_auto_mode_does_not_downgrade_an_attempted_compile_failure(tmp_path: Path) -> None:
    class FailingCompileModel(_VerifyingFakeHlsModel):
        def compile(self) -> None:
            raise RuntimeError("compiler failed")

    output_dir = tmp_path / "aria_project"

    with pytest.raises(VerificationError, match="baseline compilation"):
        optimize_project(
            FailingCompileModel(output_dir),
            config={"Profile": "aria", "Verification": {"Mode": "auto"}},
        )

    assert not output_dir.exists()


def test_auto_verification_records_a_missing_host_capability_without_compiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ravel_hls.api.inspect_dependencies",
        lambda: {
            "dependency_qualification": "qualified",
            "dependencies": {},
            "compiler": {"command": None, "status": "missing"},
            "hls_simulation_headers": {"path": "/headers/ap_fixed.h", "status": "available"},
        },
    )
    hls_model = _VerifyingFakeHlsModel(tmp_path / "aria_project")

    project = optimize_project(
        hls_model,
        config={"Profile": "aria", "Verification": {"Mode": "auto"}},
    )

    assert hls_model.compile_called is False
    assert project.status["correctness_verification"] == "not_run"
    assert "compiler" in project.manifest["verification"]["unavailable_reason"]


def test_required_verification_rejects_a_missing_host_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ravel_hls.api.inspect_dependencies",
        lambda: {
            "dependency_qualification": "qualified",
            "dependencies": {},
            "compiler": {"command": None, "status": "missing"},
            "hls_simulation_headers": {"path": None, "status": "missing"},
        },
    )
    hls_model = _VerifyingFakeHlsModel(tmp_path / "aria_project")

    with pytest.raises(VerificationError, match="compiler.*HLS simulation headers"):
        optimize_project(
            hls_model,
            config={"Profile": "aria", "Verification": {"Mode": "required"}},
        )

    assert hls_model.compile_called is False


def test_refresh_model_reuses_the_recorded_configs_through_clean_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = optimize_project(
        _FakeHlsModel(tmp_path / "original"),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    refreshed_output = tmp_path / "refreshed"
    conversion_call: dict[str, Any] = {}

    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        conversion_call.update(kwargs)
        return _FakeHlsModel(refreshed_output)

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)
    new_model = object()

    refreshed = refresh_model(original, new_model, output_dir=refreshed_output)

    assert refreshed.path == refreshed_output
    assert conversion_call["model"] is new_model
    assert conversion_call["backend"] == "Vitis"
    assert conversion_call["io_type"] == "io_stream"
    assert conversion_call["project_name"] == "aria_top"
    assert conversion_call["hls_config"]["Model"] == {
        "Strategy": "Latency",
        "ReuseFactor": 1,
    }


def test_project_refreshes_with_a_new_complete_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "project"
    original = optimize_project(
        _FakeHlsModel(output_dir),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    conversion_call: dict[str, Any] = {}

    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        conversion_call.update(kwargs)
        return _FakeHlsModel(output_dir)

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)
    new_model = object()

    refreshed = original.refresh(new_model)

    assert refreshed.path == output_dir
    assert conversion_call["model"] is new_model


def test_project_refresh_preserves_the_recorded_specialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "project"

    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        return _FakeHlsModel(Path(kwargs["output_dir"]))

    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)
    original = convert(
        object(),
        {
            "Project": {"Name": "aria_top", "OutputDir": output_dir},
            "HLS": {"Config": {"Model": {"Strategy": "Latency"}}},
            "Optimization": {"TemporalPacking": 2, "DenseParallelism": 1},
            "Verification": {"Mode": "disabled"},
        },
    )

    refreshed = original.refresh(object())

    assert refreshed.config["Optimization"] == {
        "TemporalPacking": 2,
        "DenseParallelism": 1,
    }
    assert refreshed.manifest["generation_configuration"]["ravel"][
        "Optimization"
    ] == {"TemporalPacking": 2, "DenseParallelism": 1}
    assert (
        refreshed.manifest["configuration_sha256"]
        == original.manifest["configuration_sha256"]
    )


def test_project_refreshes_from_parameters_through_the_complete_model_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "project"
    original = optimize_project(
        _FakeHlsModel(output_dir),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    parameters = Parameters.extract(_ParameterModel([4.0, 5.0]))
    template = _ParameterModel([0.0, 0.0])
    conversion_call: dict[str, Any] = {}

    def fake_load(path: Path, **kwargs: Any) -> _ParameterModel:
        assert path == output_dir / "keras_model.keras"
        return template

    def fake_convert(**kwargs: Any) -> _FakeHlsModel:
        conversion_call.update(kwargs)
        return _FakeHlsModel(output_dir)

    fake_keras = ModuleType("keras")
    fake_keras.models = SimpleNamespace(load_model=fake_load)
    fake_hgq = ModuleType("hgq")
    fake_hgq_layers = ModuleType("hgq.layers")
    fake_hgq_layers.QConv2D = object()
    fake_hgq_layers.QDense = object()
    fake_hls4ml = ModuleType("hls4ml")
    fake_hls4ml.converters = SimpleNamespace(convert_from_keras_model=fake_convert)
    monkeypatch.setitem(sys.modules, "keras", fake_keras)
    monkeypatch.setitem(sys.modules, "hgq", fake_hgq)
    monkeypatch.setitem(sys.modules, "hgq.layers", fake_hgq_layers)
    monkeypatch.setitem(sys.modules, "hls4ml", fake_hls4ml)

    refreshed = original.refresh(parameters)

    assert refreshed.path == output_dir
    assert conversion_call["model"] is template
    assert template.layers[0].weights[0].numpy().tolist() == [4.0, 5.0]


def test_project_rejects_parameters_with_a_different_quantizer_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "project"
    original = optimize_project(
        _FakeHlsModel(output_dir),
        config={"Profile": "aria", "Verification": {"Mode": "disabled"}},
    )
    parameters = Parameters.extract(_ParameterModel([4.0, 5.0], "RND"))
    template = _ParameterModel([0.0, 0.0], "TRN")

    fake_keras = ModuleType("keras")
    fake_keras.models = SimpleNamespace(load_model=lambda *args, **kwargs: template)
    fake_hgq = ModuleType("hgq")
    fake_hgq_layers = ModuleType("hgq.layers")
    fake_hgq_layers.QConv2D = object()
    fake_hgq_layers.QDense = object()
    monkeypatch.setitem(sys.modules, "keras", fake_keras)
    monkeypatch.setitem(sys.modules, "hgq", fake_hgq)
    monkeypatch.setitem(sys.modules, "hgq.layers", fake_hgq_layers)

    with pytest.raises(ConfigurationError, match="incompatible"):
        original.refresh(parameters)
