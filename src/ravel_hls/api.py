"""Primary public generation workflows."""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

from .analysis.dense import analyze_dense_facts
from .config import RavelConfig
from .compatibility.dependencies import inspect_dependencies
from .compatibility.model_profile import validate_aria_model_profile
from .backends.vitis.renderer import render_aria_project
from .backends.vitis.build import normalize_build_script, write_build_options
from .exceptions import (
    CompatibilityError,
    ConfigurationError,
    ProjectGenerationError,
    RavelError,
    VerificationError,
)
from .manifest import build_generation_manifest
from .parameters import Parameters
from .profiles.aria.plan import build_implementation_plan, build_pass_records
from .project import RavelProject, open_project
from .verification.equivalence import (
    predict_baseline,
    predict_optimized,
    prepare_stimuli,
    report_model_fidelity,
    require_bit_exact,
)


def convert(
    model: Any, config: Mapping[str, Any], *, inputs: Any | None = None
) -> RavelProject:
    """Convert a compatible model using the Aria 1.4 public configuration."""

    unknown_fields = sorted(
        config.keys() - {"Project", "HLS", "Optimization", "Verification", "Vitis"}
    )
    if unknown_fields:
        raise ConfigurationError(
            f"Unknown RAVEL configuration field: {unknown_fields[0]}"
        )
    normalized = RavelConfig(config)
    project = normalized["Project"]
    hls = normalized["HLS"]
    generated = convert_from_keras_model(
        model,
        output_dir=project["OutputDir"],
        project_name=project["Name"],
        hls_config=hls["Config"],
        ravel_config=normalized,
        backend=hls.get("Backend", "Vitis"),
        io_type=hls.get("IOType", "io_stream"),
        part=hls.get("Part"),
        clock_period=hls.get("ClockPeriod"),
        force_replace=project["ForceReplace"],
        verification_inputs=inputs,
    )
    if normalized["Vitis"]["Run"]:
        generated.build()
    return generated


def refresh_model(
    project: RavelProject | str | os.PathLike[str],
    model: Any,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    verification_inputs: Any | None = None,
    force_replace: bool = False,
) -> RavelProject:
    """Regenerate an existing RAVEL project with a complete compatible model."""

    project_view = project if isinstance(project, RavelProject) else open_project(project)
    if isinstance(model, Parameters):
        import keras
        from hgq.layers import QConv2D, QDense

        template = keras.models.load_model(
            project_view.path / "keras_model.keras",
            custom_objects={"QConv2D": QConv2D, "QDense": QDense},
        )
        model = model._apply(template)
    hls_values = _load_hls4ml_config(project_view.path / "hls4ml_config.yml")
    target = project_view.path if output_dir is None else Path(output_dir)
    return convert_from_keras_model(
        model,
        output_dir=target,
        project_name=hls_values["ProjectName"],
        hls_config=hls_values["HLSConfig"],
        ravel_config=project_view.config,
        backend=hls_values["Backend"],
        io_type=hls_values["IOType"],
        part=hls_values.get("Part"),
        clock_period=hls_values.get("ClockPeriod"),
        force_replace=force_replace,
        verification_inputs=verification_inputs,
        preserved_implementation_plan=project_view.implementation_plan,
    )


def convert_from_keras_model(
    model: Any,
    *,
    output_dir: str | os.PathLike[str],
    project_name: str,
    hls_config: Mapping[str, Any],
    ravel_config: RavelConfig | Mapping[str, Any] | None = None,
    backend: str = "Vitis",
    io_type: str = "io_stream",
    part: str | None = None,
    clock_period: float | None = None,
    force_replace: bool = False,
    verification_inputs: Any | None = None,
    preserved_implementation_plan: Mapping[str, Any] | None = None,
) -> RavelProject:
    """Convert a Keras/HGQ model and run the canonical Aria optimization engine."""

    import hls4ml

    normalized_model = model
    if isinstance(model, (str, os.PathLike)):
        import keras
        from hgq.layers import QConv2D, QDense

        model_path = Path(model)
        if not model_path.is_file():
            raise CompatibilityError(f"Keras model file does not exist: {model_path}")
        normalized_model = keras.models.load_model(
            model_path, custom_objects={"QConv2D": QConv2D, "QDense": QDense}
        )
    conversion_arguments: dict[str, Any] = {
        "model": normalized_model,
        "output_dir": str(output_dir),
        "project_name": project_name,
        "hls_config": dict(hls_config),
        "backend": backend,
        "io_type": io_type,
    }
    if part is not None:
        conversion_arguments["part"] = part
    if clock_period is not None:
        conversion_arguments["clock_period"] = clock_period
    hls_model = hls4ml.converters.convert_from_keras_model(**conversion_arguments)
    return optimize_project(
        hls_model,
        ravel_config,
        force_replace=force_replace,
        verification_inputs=verification_inputs,
        preserved_implementation_plan=preserved_implementation_plan,
    )


def optimize_project(
    hls_model: Any,
    config: RavelConfig | Mapping[str, Any] | None = None,
    *,
    force_replace: bool = False,
    verification_inputs: Any | None = None,
    preserved_implementation_plan: Mapping[str, Any] | None = None,
) -> RavelProject:
    """Generate an Aria-optimized project from a compatible hls4ml model graph."""

    ravel_config = RavelConfig(config) if not isinstance(config, RavelConfig) else config
    dependency_report = inspect_dependencies()
    if dependency_report["dependency_qualification"] != "qualified":
        failures = [
            name
            for name, facts in dependency_report["dependencies"].items()
            if facts["status"] != "qualified"
        ]
        raise CompatibilityError(
            "Aria 1.4.0 dependency stack is not qualified: " + ", ".join(failures)
        )
    if (
        verification_inputs is not None
        and ravel_config["Verification"]["Mode"] == "disabled"
    ):
        raise ConfigurationError(
            "verification_inputs cannot be supplied when Verification.Mode is disabled"
        )
    hls_config = _hls_config_values(hls_model)
    if hls_config.get("Backend") != "Vitis":
        raise CompatibilityError("hls4ml Backend must be Vitis for Aria 1.4.0")
    if hls_config.get("IOType") != "io_stream":
        raise CompatibilityError("hls4ml IOType must be io_stream for Aria 1.4.0")
    model_config = hls_config.get("HLSConfig", {}).get("Model", {})
    if model_config.get("Strategy", "Latency") != "Latency":
        raise CompatibilityError("hls4ml Strategy must be Latency for Aria 1.4.0")
    if model_config.get("ReuseFactor", 1) != 1:
        raise CompatibilityError("hls4ml ReuseFactor must be 1 for Aria 1.4.0")
    input_shapes = list(hls_config.get("InputShapes", {}).values())
    if input_shapes != [[256, 4]]:
        raise CompatibilityError(
            "Aria 1.4.0 requires one logical input shape [256, 4]"
        )
    output_shapes = list(hls_config.get("OutputShapes", {}).values())
    if output_shapes != [[1]]:
        raise CompatibilityError("Aria 1.4.0 requires one logical output shape [1]")
    layers = list(hls_model.get_layers())
    validate_aria_model_profile(layers)
    model_facts = {"dense": analyze_dense_facts(layers)}
    implementation_plan = build_implementation_plan(
        ravel_config["Optimization"], model_facts
    )
    if preserved_implementation_plan is not None:
        preserved_weight_delivery = preserved_implementation_plan.get(
            "weight_delivery"
        )
        if (
            isinstance(preserved_weight_delivery, Mapping)
            and preserved_weight_delivery.get("id") == "complete-partition"
        ):
            implementation_plan["template_profile"] = preserved_implementation_plan[
                "template_profile"
            ]
            implementation_plan["weight_delivery"] = dict(
                preserved_weight_delivery
            )
        elif (
            isinstance(preserved_weight_delivery, Mapping)
            and preserved_weight_delivery.get("id") == "wide-sequential"
        ):
            if (
                implementation_plan["template_profile"]
                != preserved_implementation_plan["template_profile"]
                or implementation_plan["weight_delivery"]
                != preserved_weight_delivery
            ):
                raise CompatibilityError(
                    "Dense implementation plan changed during refresh; "
                    "use ordinary conversion"
                )
            implementation_plan["template_profile"] = (
                preserved_implementation_plan["template_profile"]
            )
            implementation_plan["weight_delivery"] = dict(
                preserved_weight_delivery
            )
    if implementation_plan["weight_delivery"]["id"] == "wide-sequential":
        dense_applicability = implementation_plan["weight_delivery"][
            "applicability"
        ]
        if dense_applicability["status"] != "applicable":
            raise CompatibilityError("; ".join(dense_applicability["reasons"]))
    return _generate_project(
        hls_model,
        hls_config,
        ravel_config,
        layers,
        model_facts=model_facts,
        implementation_plan=implementation_plan,
        dependency_report=dependency_report,
        force_replace=force_replace,
        verification_inputs=verification_inputs,
    )


def _hls_config_values(hls_model: Any) -> Mapping[str, Any]:
    hls_config = getattr(hls_model, "config", None)
    values = getattr(hls_config, "config", None)
    if not isinstance(values, Mapping):
        raise CompatibilityError("Expected an hls4ml ModelGraph configuration")
    return values


def _generate_project(
    hls_model: Any,
    hls_config: Mapping[str, Any],
    ravel_config: RavelConfig,
    layers: list[Any],
    *,
    model_facts: Mapping[str, Any],
    implementation_plan: dict[str, Any],
    dependency_report: Mapping[str, Any],
    force_replace: bool,
    verification_inputs: Any | None,
) -> RavelProject:
    output_value = hls_config.get("OutputDir")
    if not isinstance(output_value, (str, os.PathLike)):
        raise ProjectGenerationError("hls4ml OutputDir must identify a project directory")
    output_path = Path(output_value)
    replace_existing = False
    if output_path.exists():
        try:
            open_project(output_path)
        except RavelError as error:
            if not force_replace:
                raise ProjectGenerationError(
                    f"Refusing to replace unrecognized target directory: {output_path}"
                ) from error
        replace_existing = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.ravel-", dir=output_path.parent)
    )
    mutable_hls_config = hls_model.config.config
    original_output = mutable_hls_config.get("OutputDir")
    try:
        mutable_hls_config["OutputDir"] = str(staging_path)
        hls_model.write()
        verification_mode = ravel_config["Verification"]["Mode"]
        stimuli = None
        stimuli_record = None
        baseline_predictions = None
        verification_unavailable = None
        if verification_mode != "disabled":
            stimuli, stimuli_record = prepare_stimuli(
                ravel_config, verification_inputs
            )
            verification_unavailable = _verification_unavailable_reason(
                hls_model, dependency_report
            )
            if verification_unavailable is None:
                baseline_predictions = predict_baseline(hls_model, stimuli)
            else:
                if verification_mode == "required":
                    raise VerificationError(verification_unavailable)
        pass_records = build_pass_records(implementation_plan)
        project_name = hls_config.get("ProjectName")
        if not isinstance(project_name, str) or not project_name.isidentifier():
            raise ProjectGenerationError(
                "hls4ml ProjectName must be a valid C++ identifier"
            )
        managed_paths = render_aria_project(
            staging_path,
            project_name,
            layers,
            implementation_plan=implementation_plan,
        )
        normalize_build_script(staging_path)
        write_build_options(staging_path, ravel_config)
        verification_report: dict[str, Any] = {
            "mode": verification_mode,
            "transformation_equivalence": "not_run",
            "model_fidelity": "not_run",
        }
        if stimuli_record is not None:
            verification_report["stimuli"] = stimuli_record
        if verification_unavailable is not None:
            verification_report["unavailable_reason"] = verification_unavailable
        if baseline_predictions is not None and stimuli is not None:
            optimized_predictions = predict_optimized(staging_path, stimuli)
            require_bit_exact(baseline_predictions, optimized_predictions)
            verification_report["transformation_equivalence"] = "passed"
            fidelity = report_model_fidelity(
                hls_config.get("KerasModel"), stimuli, optimized_predictions
            )
            if fidelity is not None:
                verification_report["model_fidelity"] = "reported"
                verification_report["model_fidelity_report"] = fidelity
        mutable_hls_config["OutputDir"] = original_output
        _rewrite_published_hls_config(staging_path, output_path)
        published_ravel_config = _published_ravel_config(ravel_config)
        ravel_config_path = staging_path / "ravel_config.yml"
        ravel_config_path.write_text(
            published_ravel_config.to_yaml(), encoding="utf-8"
        )
        semantic_model = {
            "facts": model_facts,
            "layers": [
                {
                    "class_name": layer.class_name,
                    "attributes": _semantic_attributes(layer),
                    "parameters": [
                        _semantic_parameter(weight) for weight in layer.get_weights()
                    ],
                }
                for layer in layers
            ]
        }
        normalized_hls_config = _normalized_hls_config(hls_config)
        manifest = build_generation_manifest(
            project_path=staging_path,
            hls_config=normalized_hls_config,
            ravel_config=published_ravel_config,
            semantic_model=semantic_model,
            implementation_plan=implementation_plan,
            pass_records=pass_records,
            verification_report=verification_report,
            interface_contract=_interface_contract(layers, implementation_plan),
        )
        (staging_path / "ravel_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if replace_existing:
            backup_path = output_path.parent / (
                f".{output_path.name}.ravel-backup-{uuid.uuid4().hex}"
            )
            output_path.rename(backup_path)
            try:
                staging_path.rename(output_path)
            except Exception:
                backup_path.rename(output_path)
                raise
            shutil.rmtree(backup_path)
        else:
            staging_path.rename(output_path)
    except Exception:
        mutable_hls_config["OutputDir"] = original_output
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise
    return open_project(output_path)


def _verification_unavailable_reason(
    hls_model: Any, dependency_report: Mapping[str, Any]
) -> str | None:
    missing = []
    if not callable(getattr(hls_model, "compile", None)) or not callable(
        getattr(hls_model, "predict", None)
    ):
        missing.append("hls4ml compile/predict capability")
    if dependency_report.get("compiler", {}).get("status") != "available":
        missing.append("C++ compiler")
    if (
        dependency_report.get("hls_simulation_headers", {}).get("status")
        != "available"
    ):
        missing.append("HLS simulation headers")
    if not missing:
        return None
    return "Required verification capability is unavailable: " + ", ".join(missing)


def _semantic_attributes(layer: Any) -> dict[str, Any]:
    names = (
        "target_shape",
        "in_height",
        "in_width",
        "n_chan",
        "filt_height",
        "filt_width",
        "n_filt",
        "stride_height",
        "stride_width",
        "pad_top",
        "pad_bottom",
        "pad_left",
        "pad_right",
        "out_height",
        "out_width",
        "activation",
        "n_in",
        "pool_height",
        "pool_width",
        "pool_op",
        "n_out",
    )
    return {
        name: layer.get_attr(name)
        for name in names
        if layer.get_attr(name) is not None
    }


def _semantic_parameter(weight: Any) -> dict[str, Any]:
    import numpy as np

    values = np.ascontiguousarray(weight.data)
    return {
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "values_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "precision": weight.type.precision.definition_cpp(),
    }


def _normalized_hls_config(hls_config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = hls_config.get("HLSConfig", {}).get("Model", {})
    return {
        "Backend": hls_config.get("Backend"),
        "IOType": hls_config.get("IOType"),
        "ProjectName": hls_config.get("ProjectName"),
        "Part": hls_config.get("Part"),
        "ClockPeriod": hls_config.get("ClockPeriod"),
        "Strategy": model_config.get("Strategy", "Latency"),
        "ReuseFactor": model_config.get("ReuseFactor", 1),
    }


def _interface_contract(
    layers: list[Any], implementation_plan: Mapping[str, Any]
) -> dict[str, Any]:
    input_variable = layers[0].get_output_variable()
    output_variable = layers[-1].get_output_variable()
    input_precision = input_variable.type.precision
    output_precision = output_variable.type.precision
    input_width = getattr(input_precision, "width", None)
    output_width = getattr(output_precision, "width", None)
    if not isinstance(input_width, int) or not isinstance(output_width, int):
        raise ProjectGenerationError("Unable to resolve Aria interface precision widths")
    input_slot_width = max(8, 1 << (input_width - 1).bit_length())
    output_slot_width = max(8, 1 << (output_width - 1).bit_length())
    return {
        "logical_model_interface": {
            "input_shape": [256, 4],
            "output_shape": [1],
        },
        "hls_stream_interface": {
            "input_rows_per_word": implementation_plan["temporal_pack"],
            "channels_per_row": 4,
            "values_per_input_word": implementation_plan["values_per_input_word"],
            "input_words_per_inference": implementation_plan[
                "input_words_per_inference"
            ],
            "output_words_per_inference": 1,
            "input_scalar_bits": input_width,
            "output_scalar_bits": output_width,
            "ordering": "row-major; time before channel",
            "protocol": "axis",
            "block_control": "ap_ctrl_hs",
            "optional_axis_sidebands": [],
        },
        "rtl_interface": {
            "expected": {
                "qualification_profile": (
                    "hls4ml-1.2.0-vitis-2023.2-axis-packing-v1"
                ),
                "input_tdata_bits": (
                    implementation_plan["values_per_input_word"] * input_slot_width
                ),
                "output_tdata_bits": output_slot_width,
                "input_tdata_port": f"{input_variable.name}_TDATA",
                "output_tdata_port": f"{output_variable.name}_TDATA",
                "input_scalar_bits": input_width,
                "output_scalar_bits": output_width,
            },
            "measured": None,
        },
    }


class _KerasModelPath(str):
    pass


def _rewrite_published_hls_config(staging_path: Path, output_path: Path) -> None:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    class Dumper(yaml.SafeDumper):
        pass

    Loader.add_constructor(
        "!keras_model",
        lambda loader, node: _KerasModelPath(loader.construct_scalar(node)),
    )
    Dumper.add_representer(
        _KerasModelPath,
        lambda dumper, value: dumper.represent_scalar(
            "!keras_model", str(value), style="'"
        ),
    )
    config_path = staging_path / "hls4ml_config.yml"
    try:
        values = yaml.load(config_path.read_text(encoding="utf-8"), Loader=Loader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ProjectGenerationError(
            f"Cannot normalize hls4ml_config.yml for publication: {error}"
        ) from error
    if not isinstance(values, dict):
        raise ProjectGenerationError("hls4ml_config.yml must contain a mapping")
    values["OutputDir"] = "."
    if isinstance(values.get("KerasModel"), _KerasModelPath):
        values["KerasModel"] = _KerasModelPath("keras_model.keras")
    config_path.write_text(
        yaml.dump(values, Dumper=Dumper, sort_keys=False), encoding="utf-8"
    )


def _load_hls4ml_config(config_path: Path) -> dict[str, Any]:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor(
        "!keras_model", lambda loader, suffix, node: loader.construct_scalar(node)
    )
    try:
        values = yaml.load(config_path.read_text(encoding="utf-8"), Loader=Loader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ProjectGenerationError(
            f"Cannot read recorded hls4ml configuration: {error}"
        ) from error
    if not isinstance(values, dict):
        raise ProjectGenerationError("Recorded hls4ml configuration must be a mapping")
    return values


def _published_ravel_config(config: RavelConfig) -> RavelConfig:
    values = config.to_dict()
    project = values.get("Project")
    if isinstance(project, dict):
        project["OutputDir"] = "."
        return RavelConfig(values)
    return config
