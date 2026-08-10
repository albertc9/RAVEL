# RAVEL

RAVEL (Rate-Aware Vectorized Engine for Low-latency) generates a specialized,
hls4ml-compatible FPGA inference project. Aria 1.1.0 implements the qualified
pair-parallel, two-row wide-stream profile for the CNN-for-Arianna model family;
it is not a design-space explorer and it does not impose application-specific
performance targets.

## Install

Use a clean Python 3.11 environment on Linux for the fully qualified generation,
C++ verification, and Vitis HLS workflow:

```bash
python -m pip install ravel-hls==1.1.0
ravel-hls doctor --json
```

For an editable source checkout, replace the install command with:

```bash
python -m pip install -c constraints/aria-reference.txt -e .
```

Do not co-install the retired `HGQ` distribution with `hgq2`; both own the
`hgq` Python namespace and RAVEL rejects that conflict before generation.

## Python API

```python
import hls4ml
import keras
from hgq.layers import QConv2D, QDense
import ravel_hls as ravel

model = keras.models.load_model(
    "model.keras", custom_objects={"QConv2D": QConv2D, "QDense": QDense}
)
hls = hls4ml.utils.config_from_keras_model(
    model, granularity="name", backend="Vitis"
)
hls["Model"].update({"Strategy": "Latency", "ReuseFactor": 1})

config = {
    "Project": {"Name": "cnn_core", "OutputDir": "cnn_core"},
    "HLS": {
        "Backend": "Vitis",
        "IOType": "io_stream",
        "Part": "xcku5p-ffvb676-2-e",
        "ClockPeriod": 5.0,
        "Config": hls,
    },
    "Verification": {"Mode": "required", "Samples": 32, "Seed": 19},
    "Vitis": {"Run": False},
}

project = ravel.convert(model, config)
print(project.status)
```

`Vitis.Run` defaults to `False`. Set it to `True` to run
`vitis_hls -f build_prj.tcl` after atomic project publication and automatically
record the synthesis report. The default Vitis stages are reset and synthesis;
CSim, CoSim, validation, export, and Vivado synthesis remain disabled unless
their booleans under `Vitis.Stages` are enabled explicitly. The same operation
can be requested later with `project.build()`.

The concise project lifecycle is `Project.open(path)`, `project.refresh(model)`,
`project.build()`, `project.record(report_dir)`, and `project.link()`. The CLI
command `ravel-hls inspect PROJECT --json` performs full source-integrity
checking; add `--fast` when payload hashing should be skipped.

## Parameter packages

`Parameters` stores portable generation-relevant inference state without
generated HLS sources or executable Python objects:

```python
parameters = ravel.Parameters.extract(model)
parameters.save("trained.ravelparams")

project = ravel.Project.open("cnn_core")
project.refresh(ravel.Parameters.load("trained.ravelparams"))
```

The deterministic archive contains JSON plus NPY arrays for kernel, bias, and
learned K/I/F quantizer state. Static quantizer contracts and slot schemas are
compatibility-checked before a complete staged regeneration. A parameter
package is not encrypted; treat it as sensitive model IP.

## Evidence boundary

Generation, correctness, model fidelity, source integrity, and vendor
measurements are independent status axes. Qualification records measured II,
latency, timing, resources, and RTL ports for one exact generation fingerprint;
it does not compare those measurements with an application threshold. A high II
or an estimated clock above the requested clock remains recorded evidence, not
a failed RAVEL project.

See the executable [CNN-for-Arianna reference](references/cnn_for_arianna/README.md),
[architecture](docs/architecture.md), [compatibility](docs/compatibility.md), and
[project format](docs/project-format.md) for the full contracts.
