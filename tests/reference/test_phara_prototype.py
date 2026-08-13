from copy import deepcopy
from pathlib import Path

import numpy as np

from ravel_hls.analysis.model import _analyze_model
from ravel_hls.compatibility.dependencies import inspect_dependencies
from ravel_hls.profiles.aria.plan import build_implementation_plan
from ravel_hls.rendering.vitis import render_aria_project
from ravel_hls.verification.equivalence import (
    predict_baseline,
    predict_optimized,
    require_bit_exact,
)


REFERENCE_MODEL = (
    Path(__file__).parents[2]
    / "references"
    / "cnn_for_arianna"
    / "models"
    / "cnn_for_arianna.keras"
)


def test_phara_p4_direct_fused_cpp_is_bit_exact(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    analyzed = _analyze_model(
        REFERENCE_MODEL,
        {
            "HLS": {"Backend": "Vitis", "IOType": "io_stream"},
            "Optimization": {"TemporalPacking": 4, "DenseParallelism": 2},
        },
    )
    report = analyzed.analysis.to_dict()
    project_path = tmp_path / "phara_p4_direct"
    analyzed.graph.config.config["OutputDir"] = str(project_path)
    analyzed.graph.write()
    compiler = inspect_dependencies()["compiler"]["command"]
    inputs = np.zeros((3, 256, 4), dtype=np.float32)
    inputs[1].fill(1.0)
    inputs[2] = np.random.default_rng(7).integers(
        -32, 33, size=(256, 4)
    ).astype(np.float32) / 32.0
    baseline = predict_baseline(analyzed.graph, inputs, compiler)

    resolved_design = deepcopy(report["resolved_design"])
    resolved_design["implementation_plan"] = build_implementation_plan(
        {"TemporalPacking": 4, "DenseParallelism": 4},
        report["model_facts"],
    )
    resolved_design["specialization"]["dense_parallelism"] = 4
    render_aria_project(
        project_path,
        analyzed.graph.config.config["ProjectName"],
        resolved_design,
        analyzed.parameter_payload,
    )

    top_source = (project_path / "firmware" / "ravel_analysis.cpp").read_text(
        encoding="utf-8"
    )
    assert "phara_pool_aligned_direct_p4_cl" in top_source
    optimized = predict_optimized(project_path, inputs, compiler)
    require_bit_exact(baseline, optimized)
