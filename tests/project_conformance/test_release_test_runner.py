from pathlib import Path
import subprocess
import sys


def test_release_test_runner_lists_only_the_curated_release_smoke_gate() -> None:
    repository = Path(__file__).resolve().parents[2]
    runner = repository / ".github/scripts/run_release_tests.py"
    completed = subprocess.run(
        [sys.executable, str(runner), "--list"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    expected = [
        (
            "tests/project_conformance/test_distribution.py::"
            "test_release_build_uses_the_dedicated_runner_without_moving_pypi_publish"
        ),
        (
            "tests/project_conformance/test_distribution.py::"
            "test_public_namespace_exposes_only_the_aria_1_5_lifecycle"
        ),
        (
            "tests/project_conformance/test_distribution.py::"
            "test_wheel_contains_aria_rendering_templates"
        ),
        "tests/project_conformance/test_import_boundaries.py",
        "tests/project_conformance/test_release_test_runner.py",
        "tests/project_conformance/test_release_version.py",
        (
            "tests/qualification/vitis/test_import.py::"
            "test_import_records_first_convolution_pipeline_evidence"
        ),
        "tests/reference/test_aria_1_5_1_rtl_evidence.py",
        (
            "tests/reference/test_dynamic_conversion.py::"
            "test_conversion_checks_implementation_consistency_without_accuracy_labels"
        ),
        "tests/unit/test_generation_registry.py",
    ]
    assert completed.stdout.splitlines() == expected
