"""Primary public generation workflows."""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

from .config import RavelConfig
from .compatibility.dependencies import inspect_dependencies
from .backends.vitis.build import normalize_build_script, write_build_options
from .exceptions import (
    CompatibilityError,
    ConfigurationError,
    ProjectGenerationError,
    RavelError,
    VerificationError,
)
from .manifest import architecture_contract_sha256, build_generation_manifest
from .parameters import Parameters
from .project import RavelProject, open_project
from .generations import builtin_generation
from .verification.equivalence import (
    predict_baseline,
    predict_optimized,
    prepare_stimuli,
    report_model_fidelity,
    require_bit_exact,
    require_source_consistency,
)


def convert(
    model: Any,
    output_dir: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    verification_inputs: Any | None = None,
) -> RavelProject:
    """Convert a compatible model using one Aria configuration."""

    return _convert_analyzed_model(
        model,
        Path(output_dir),
        config,
        verification_inputs=verification_inputs,
    )


def _convert_analyzed_model(
    model: Any,
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    verification_inputs: Any | None,
) -> RavelProject:
    from .analysis.model import _analyze_model

    unknown_fields = sorted(
        config.keys() - {"Project", "HLS", "Optimization", "Verification", "Vitis"}
    )
    if unknown_fields:
        raise ConfigurationError(
            f"Unknown RAVEL configuration field: {unknown_fields[0]}"
        )
    analyzed = _analyze_model(model, config)
    if not analyzed.analysis.applicable:
        messages = [finding["message"] for finding in analyzed.analysis.findings]
        raise CompatibilityError("; ".join(messages))
    return _publish_analyzed_graph(
        graph=analyzed.graph,
        analysis_report=analyzed.analysis.to_dict(),
        parameter_payload=analyzed.parameter_payload,
        output_dir=output_dir,
        config=config,
        verification_inputs=verification_inputs,
        source_consistency_available=True,
    )


def _publish_analyzed_graph(
    *,
    graph: Any,
    analysis_report: Mapping[str, Any],
    parameter_payload: Any,
    output_dir: Path,
    config: Mapping[str, Any],
    verification_inputs: Any | None,
    source_consistency_available: bool,
) -> RavelProject:
    hls = config.get("HLS", {})
    project = config.get("Project", {})
    if not isinstance(project, Mapping):
        raise ConfigurationError("Project must be a mapping")
    force_replace = project.get("ForceReplace", False)
    if not isinstance(force_replace, bool):
        raise ConfigurationError("Project.ForceReplace must be a boolean")
    project_name = output_dir.name
    if not project_name.isidentifier():
        raise ConfigurationError(
            "output_dir name must be a valid C++ project identifier"
        )
    graph_config = graph.config.config
    graph_config["OutputDir"] = str(output_dir)
    graph_config["ProjectName"] = project_name
    run_config = RavelConfig(
        {
            "Project": {
                "Name": project_name,
                "OutputDir": str(output_dir),
                "ForceReplace": force_replace,
            },
            "HLS": {
                "Backend": hls.get("Backend", "Vitis"),
                "IOType": hls.get("IOType", "io_stream"),
                "Part": hls.get("Part"),
                "ClockPeriod": hls.get("ClockPeriod"),
                "Config": deepcopy(graph_config["HLSConfig"]),
            },
            "Optimization": dict(config.get("Optimization", {})),
            "Verification": dict(config.get("Verification", {})),
            "Vitis": dict(config.get("Vitis", {})),
        }
    )
    generated = _generate_analyzed_project(
        graph,
        run_config,
        force_replace=force_replace,
        verification_inputs=verification_inputs,
        model_analysis=analysis_report,
        parameter_payload=parameter_payload,
        source_consistency_available=source_consistency_available,
    )
    if run_config["Vitis"]["Run"]:
        generated.build()
    return generated


def refresh(
    project: RavelProject | str | os.PathLike[str],
    model_or_parameters: Any,
    *,
    verification_inputs: Any | None = None,
) -> RavelProject:
    """Atomically refresh a schema-v4 project without changing its architecture."""

    from .analysis.model import _analyze_model, analyze

    project_view = project if isinstance(project, RavelProject) else open_project(project)
    if project_view.manifest.get("schema_version") != 4:
        raise CompatibilityError(
            "Aria 1.5 refresh requires a schema-v4 generated project"
        )
    if isinstance(model_or_parameters, Parameters):
        config = _refresh_configuration(project_view)
        if (
            config["Verification"]["Mode"] == "required"
            and model_or_parameters._manifest.get("known_answer_evidence") is None
        ):
            raise VerificationError(
                "Required package refresh verification needs known-answer evidence"
            )
        if (
            model_or_parameters.model_structure_sha256
            != project_view.manifest["source_model"]["fingerprints"][
                "model_structure_sha256"
            ]
        ):
            raise CompatibilityError(
                "Parameter package changes the recorded architecture contract; "
                "use ordinary conversion"
            )
        import keras
        from hgq.layers import QConv2D, QDense

        template = keras.models.load_model(
            project_view.path / "keras_model.keras",
            custom_objects={"QConv2D": QConv2D, "QDense": QDense},
        )
        analyzed = _analyze_model(template, config)
        payload, report = model_or_parameters._apply_to_analysis(analyzed)
        if architecture_contract_sha256(report) != project_view.manifest.get(
            "architecture_contract_sha256"
        ):
            raise CompatibilityError(
                "Parameter package changes the recorded architecture contract; "
                "use ordinary conversion"
            )
        return _publish_analyzed_graph(
            graph=analyzed.graph,
            analysis_report=report,
            parameter_payload=payload,
            output_dir=project_view.path,
            config=config,
            verification_inputs=verification_inputs,
            source_consistency_available=False,
        )
    config = _refresh_configuration(project_view)
    analysis = analyze(model_or_parameters, config).to_dict()
    observed_contract = architecture_contract_sha256(analysis)
    expected_contract = project_view.manifest.get("architecture_contract_sha256")
    if observed_contract != expected_contract:
        raise CompatibilityError(
            "Refresh model changes the recorded architecture contract; "
            "use ordinary conversion"
        )
    return convert(
        model_or_parameters,
        project_view.path,
        config,
        verification_inputs=verification_inputs,
    )


def _refresh_configuration(project: RavelProject) -> dict[str, Any]:
    recorded = project.config.to_dict()
    hls = recorded["HLS"]
    return {
        "Project": {"ForceReplace": True},
        "HLS": {
            "Backend": hls["Backend"],
            "IOType": hls["IOType"],
            "Part": hls.get("Part"),
            "ClockPeriod": hls.get("ClockPeriod"),
        },
        "Optimization": recorded["Optimization"],
        "Verification": recorded["Verification"],
        "Vitis": recorded["Vitis"],
    }


def _generate_analyzed_project(
    hls_model: Any,
    config: RavelConfig,
    *,
    force_replace: bool = False,
    verification_inputs: Any | None = None,
    model_analysis: Mapping[str, Any],
    parameter_payload: Any,
    source_consistency_available: bool = True,
) -> RavelProject:
    """Generate from the graph-bearing result of the public analysis path."""

    ravel_config = config
    dependency_report = inspect_dependencies()
    if dependency_report["dependency_qualification"] != "qualified":
        failures = [
            name
            for name, facts in dependency_report["dependencies"].items()
            if facts["status"] != "qualified"
        ]
        raise CompatibilityError(
            "Aria 1.5.1 dependency stack is not qualified: " + ", ".join(failures)
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
        raise CompatibilityError("hls4ml Backend must be Vitis for Aria 1.5.1")
    if hls_config.get("IOType") != "io_stream":
        raise CompatibilityError("hls4ml IOType must be io_stream for Aria 1.5.1")
    model_config = hls_config.get("HLSConfig", {}).get("Model", {})
    if model_config.get("Strategy", "Latency") != "Latency":
        raise CompatibilityError("hls4ml Strategy must be Latency for Aria 1.5.1")
    if model_config.get("ReuseFactor", 1) != 1:
        raise CompatibilityError("hls4ml ReuseFactor must be 1 for Aria 1.5.1")
    layers = list(hls_model.get_layers())
    input_shapes = list(hls_config.get("InputShapes", {}).values())
    output_shapes = list(hls_config.get("OutputShapes", {}).values())
    model_facts = dict(model_analysis["model_facts"])
    expected_input = model_facts["operations"][0]["outputs"][0]["shape"]
    expected_output = model_facts["operations"][-1]["outputs"][0]["shape"]
    if input_shapes != [expected_input] or output_shapes != [expected_output]:
        raise CompatibilityError(
            "Analyzed model facts disagree with the hls4ml graph interface"
        )
    implementation_plan = deepcopy(
        model_analysis["resolved_design"]["implementation_plan"]
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
        model_analysis=model_analysis,
        parameter_payload=parameter_payload,
        source_consistency_available=source_consistency_available,
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
    model_analysis: Mapping[str, Any],
    parameter_payload: Any,
    source_consistency_available: bool,
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
                ravel_config,
                verification_inputs,
                model_analysis["model_facts"]["operations"][0]["outputs"][0],
            )
            verification_unavailable = _verification_unavailable_reason(
                hls_model, dependency_report
            )
            if verification_unavailable is None:
                baseline_predictions = predict_baseline(
                    hls_model,
                    stimuli,
                    dependency_report.get("compiler", {}).get("command"),
                )
            else:
                if verification_mode == "required":
                    raise VerificationError(verification_unavailable)
        pass_records = deepcopy(
            model_analysis["resolved_design"]["executed_passes"]
        )
        project_name = hls_config.get("ProjectName")
        if not isinstance(project_name, str) or not project_name.isidentifier():
            raise ProjectGenerationError(
                "hls4ml ProjectName must be a valid C++ identifier"
            )
        generation = builtin_generation(
            model_analysis["generation"]["id"],
            model_analysis["generation"]["version"],
        )
        binding = generation.backend_binding(
            hls_config["Backend"], hls_config["IOType"]
        )
        managed_paths = binding.render(
            staging_path,
            project_name,
            model_analysis["resolved_design"],
            parameter_payload,
        )
        normalize_build_script(staging_path)
        write_build_options(staging_path, ravel_config)
        verification_report: dict[str, Any] = {
            "mode": verification_mode,
            "source_conversion_consistency": "not_run",
            "transformation_equivalence": "not_run",
            "model_fidelity": "not_run",
        }
        if stimuli_record is not None:
            verification_report["stimuli"] = stimuli_record
        if verification_unavailable is not None:
            verification_report["unavailable_reason"] = verification_unavailable
        if baseline_predictions is not None and stimuli is not None:
            optimized_predictions = predict_optimized(
                staging_path,
                stimuli,
                dependency_report.get("compiler", {}).get("command"),
            )
            require_bit_exact(baseline_predictions, optimized_predictions)
            verification_report["transformation_equivalence"] = "passed"
            if source_consistency_available:
                source_consistency = require_source_consistency(
                    hls_config.get("KerasModel"),
                    stimuli,
                    baseline_predictions,
                    model_analysis["model_facts"]["operations"][-1]["outputs"][0][
                        "numeric_type"
                    ],
                )
                verification_report["source_conversion_consistency"] = "passed"
                verification_report["source_conversion_report"] = source_consistency
            if source_consistency_available:
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
            model_analysis=dict(model_analysis),
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
    input_shape = [
        int(dimension)
        for dimension in getattr(input_variable, "shape", (256, 4))
    ]
    output_shape = [
        int(dimension) for dimension in getattr(output_variable, "shape", (1,))
    ]
    output_values = 1
    for dimension in output_shape:
        output_values *= dimension
    return {
        "logical_model_interface": {
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
        "hls_stream_interface": {
            "input_rows_per_word": implementation_plan["temporal_pack"],
            "channels_per_row": implementation_plan["channels_per_row"],
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
            "output_tdata_bits": output_values * output_slot_width,
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


def _published_ravel_config(config: RavelConfig) -> RavelConfig:
    values = config.to_dict()
    project = values.get("Project")
    if isinstance(project, dict):
        project["OutputDir"] = "."
        return RavelConfig(values)
    return config
