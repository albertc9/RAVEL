#!/usr/bin/env python3
"""Run the curated release smoke gate with bounded process memory."""

import argparse
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
RELEASE_TESTS = (
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
)


def run_release_tests(repository: Path, tests: tuple[str, ...]) -> int:
    """Run the release tests sequentially, releasing state between groups."""

    for test in tests:
        print(f"\n== {test} ==", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", test],
            cwd=repository,
            check=False,
        )
        if completed.returncode != 0:
            return completed.returncode if completed.returncode > 0 else 1
    return 0


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        action="store_true",
        help="list discovered test modules without running them",
    )
    options = parser.parse_args(arguments)
    if options.list:
        for test in RELEASE_TESTS:
            print(test)
        return 0
    return run_release_tests(REPOSITORY, RELEASE_TESTS)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
