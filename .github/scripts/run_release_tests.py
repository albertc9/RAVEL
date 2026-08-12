#!/usr/bin/env python3
"""Run each test module in a fresh process to bound release-runner memory."""

import argparse
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]


def discover_test_modules(repository: Path) -> tuple[Path, ...]:
    """Return every pytest module in deterministic repository-relative order."""

    return tuple(
        sorted(
            path.relative_to(repository)
            for path in (repository / "tests").rglob("test_*.py")
            if path.is_file()
        )
    )


def run_test_modules(repository: Path, modules: tuple[Path, ...]) -> int:
    """Run all modules sequentially, releasing interpreter state between them."""

    for module in modules:
        print(f"\n== {module.as_posix()} ==", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", module.as_posix()],
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
    modules = discover_test_modules(REPOSITORY)
    if options.list:
        for module in modules:
            print(module.as_posix())
        return 0
    return run_test_modules(REPOSITORY, modules)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
