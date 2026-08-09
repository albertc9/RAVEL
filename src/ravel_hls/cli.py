"""Command-line adapter for RAVEL."""

import argparse
from collections.abc import Sequence
from importlib.metadata import version


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RAVEL command-line interface."""

    parser = argparse.ArgumentParser(prog="ravel-hls")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('ravel-hls')} (RAVEL Aria 1.0)",
    )
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
