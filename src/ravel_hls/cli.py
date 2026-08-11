"""Command-line adapter for RAVEL."""

import argparse
from collections.abc import Sequence
from importlib.metadata import version
import json
import sys

from .compatibility.dependencies import inspect_dependencies
from .exceptions import RavelError
from .project import open_project


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RAVEL command-line interface."""

    parser = argparse.ArgumentParser(prog="ravel-hls")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('ravel-hls')} (RAVEL Aria 1.3.0)",
    )
    subparsers = parser.add_subparsers(dest="command")
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--json", action="store_true")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("project_directory")
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.add_argument("--fast", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "doctor":
        report = inspect_dependencies()
        if arguments.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(f"Dependency qualification: {report['dependency_qualification']}")
        return 0 if report["dependency_qualification"] == "qualified" else 1
    if arguments.command == "inspect":
        try:
            project = open_project(arguments.project_directory)
        except RavelError as error:
            print(f"ravel-hls: error: {error}", file=sys.stderr)
            return 2
        status = project._status(check_integrity=not arguments.fast)
        report = {
            "project": str(project.path),
            "ravel": project.manifest["ravel"],
            "status": status,
        }
        if arguments.json:
            print(json.dumps(report, sort_keys=True))
        else:
            for label, key in (
                ("Generation", "generation"),
                ("Dependency qualification", "dependency_qualification"),
                ("Correctness verification", "correctness_verification"),
                ("Model fidelity", "model_fidelity"),
                ("Source integrity", "source_integrity"),
                ("Performance qualification", "performance_qualification"),
            ):
                print(f"{label}: {status[key]}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
