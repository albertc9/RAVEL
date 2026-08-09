# RAVEL

RAVEL (Rate-Aware Vectorized Engine for Low-latency) generates a specialized,
hls4ml-compatible FPGA inference project for the fixed CNN-for-Arianna model
family. Aria 1.0 implements the pair-parallel, two-row wide-stream design; it is
not a design-space explorer.

## Install

Use a clean Python 3.11 environment on Linux for the fully qualified generation
and C++ verification workflow:

```bash
python -m pip install -c constraints/aria-reference.txt -e '.[reference]'
ravel-hls doctor --json
```

The dependency profile is intentionally exact. Do not co-install the retired
`HGQ` distribution with `hgq2`; both own the `hgq` Python namespace and RAVEL
will reject the conflict before generation.

## Python API

```python
import hls4ml
import keras
from hgq.layers import QConv2D, QDense
from ravel_hls import RavelConfig, convert_from_keras_model

model = keras.models.load_model(
    "model.keras", custom_objects={"QConv2D": QConv2D, "QDense": QDense}
)
hls_config = hls4ml.utils.config_from_keras_model(
    model, granularity="name", backend="Vitis"
)
hls_config["Model"].update({"Strategy": "Latency", "ReuseFactor": 1})

project = convert_from_keras_model(
    model,
    output_dir="cnn_core",
    project_name="cnn_core",
    hls_config=hls_config,
    ravel_config=RavelConfig(
        {"Verification": {"Mode": "required", "Samples": 32, "Seed": 19}}
    ),
    part="xcku5p-ffvb676-2-e",
    clock_period=5.0,
)
print(project.status)
```

The public lifecycle also provides `optimize_project`, `open_project`,
`refresh_model`, `RavelProject.link_hls4ml`, and `import_vitis_reports`.
Generated projects are staged and atomically published only after enabled
checks pass. `ravel-hls inspect PROJECT --json` recomputes managed-source
integrity without modifying the project.

## Reference and evidence boundary

[`references/cnn_for_arianna`](references/cnn_for_arianna/README.md) contains
the canonical model and executable reference workflow. Its `legacy/` directory
preserves selected retired-generator sources and a historical Vitis report for
comparison only; production code never imports them.

Aria generation and bit-exact baseline/optimized C++ equivalence are qualified
on the pinned Linux stack. Vitis synthesis, initiation interval, timing,
resources, and measured RTL ports are separate evidence. A report is recorded
only when `import_vitis_reports` links Vitis 2023.2 results to the exact current
manifest and its expected stream widths.

See [architecture](docs/architecture.md),
[compatibility](docs/compatibility.md), and
[project format](docs/project-format.md) for the precise contracts.
