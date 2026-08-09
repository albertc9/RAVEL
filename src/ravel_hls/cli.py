"""Command-line adapter for RAVEL."""

import argparse
from collections.abc import Sequence
from importlib.metadata import version
import json

from .compatibility.dependencies import inspect_dependencies


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RAVEL command-line interface."""

    parser = argparse.ArgumentParser(prog="ravel-hls")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('ravel-hls')} (RAVEL Aria 1.0)",
    )
    subparsers = parser.add_subparsers(dest="command")
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "doctor":
        report = inspect_dependencies()
        if arguments.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(f"Dependency qualification: {report['dependency_qualification']}")
        return 0 if report["dependency_qualification"] == "qualified" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
