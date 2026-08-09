"""Primary public generation workflows."""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

from .config import RavelConfig
from .compatibility.model_profile import validate_aria_model_profile
from .backends.vitis.renderer import render_aria_project
from .exceptions import CompatibilityError, ProjectGenerationError, RavelError
from .manifest import build_generation_manifest
from .profiles.aria.plan import build_implementation_plan, build_pass_records
from .project import RavelProject, open_project


def optimize_project(
    hls_model: Any,
    config: RavelConfig | Mapping[str, Any] | None = None,
    *,
    force_replace: bool = False,
) -> RavelProject:
    """Generate an Aria-optimized project from a compatible hls4ml model graph."""

    ravel_config = RavelConfig(config) if not isinstance(config, RavelConfig) else config
    hls_config = _hls_config_values(hls_model)
    if hls_config.get("Backend") != "Vitis":
        raise CompatibilityError("hls4ml Backend must be Vitis for Aria 1.0")
    if hls_config.get("IOType") != "io_stream":
        raise CompatibilityError("hls4ml IOType must be io_stream for Aria 1.0")
    model_config = hls_config.get("HLSConfig", {}).get("Model", {})
    if model_config.get("Strategy", "Latency") != "Latency":
        raise CompatibilityError("hls4ml Strategy must be Latency for Aria 1.0")
    if model_config.get("ReuseFactor", 1) != 1:
        raise CompatibilityError("hls4ml ReuseFactor must be 1 for Aria 1.0")
    input_shapes = list(hls_config.get("InputShapes", {}).values())
    if input_shapes != [[256, 4]]:
        raise CompatibilityError(
            "Aria 1.0 requires one logical input shape [256, 4]"
        )
    output_shapes = list(hls_config.get("OutputShapes", {}).values())
    if output_shapes != [[1]]:
        raise CompatibilityError("Aria 1.0 requires one logical output shape [1]")
    layers = list(hls_model.get_layers())
    validate_aria_model_profile(layers)
    return _generate_project(
        hls_model, hls_config, ravel_config, layers, force_replace=force_replace
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
    force_replace: bool,
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
        mutable_hls_config["OutputDir"] = original_output
        implementation_plan = build_implementation_plan()
        pass_records = build_pass_records()
        project_name = hls_config.get("ProjectName")
        if not isinstance(project_name, str) or not project_name.isidentifier():
            raise ProjectGenerationError(
                "hls4ml ProjectName must be a valid C++ identifier"
            )
        managed_paths = render_aria_project(
            staging_path, project_name, layers
        )
        ravel_config_path = staging_path / "ravel_config.yml"
        ravel_config_path.write_text(ravel_config.to_yaml(), encoding="utf-8")
        semantic_model = {
            "layers": [
                {
                    "class_name": layer.class_name,
                    "attributes": _semantic_attributes(layer),
                }
                for layer in layers
            ]
        }
        normalized_hls_config = _normalized_hls_config(hls_config)
        manifest = build_generation_manifest(
            project_path=staging_path,
            hls_config=normalized_hls_config,
            ravel_config=ravel_config,
            semantic_model=semantic_model,
            implementation_plan=implementation_plan,
            pass_records=pass_records,
            managed_paths=[*managed_paths, "ravel_config.yml"],
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
