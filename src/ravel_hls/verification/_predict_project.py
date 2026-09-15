"""Fresh-process adapter for verifying a rendered hls4ml project."""

from pathlib import Path
import sys

import numpy as np


def main() -> int:
    project_path = Path(sys.argv[1])
    input_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    from hls4ml.utils.link import FilesystemModelGraph

    try:
        linked = FilesystemModelGraph(project_path)
        linked.config.config["OutputDir"] = str(project_path.resolve())
        linked.compile()
    except Exception as error:
        print(error, file=sys.stderr)
        return 10
    try:
        inputs = np.load(input_path, allow_pickle=False)
        outputs = np.asarray(linked.predict(inputs))
        np.save(output_path, outputs, allow_pickle=False)
    except Exception as error:
        print(error, file=sys.stderr)
        return 11
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
