import hashlib
import os
from pathlib import Path
import subprocess
import sys


REFERENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "references" / "cnn_for_arianna"
)


def test_reference_model_has_the_migrated_canonical_identity() -> None:
    model_path = REFERENCE_ROOT / "models" / "cnn_for_arianna.keras"

    assert model_path.is_file()
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == (
        "65021d84030d9c09a7f1fd541221b150dad14858ad85458912a1a6a6b40a9978"
    )


def test_reference_generator_exposes_a_lightweight_help_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, str(REFERENCE_ROOT / "generate.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Generate the CNN-for-Arianna Aria project" in result.stdout
    assert "--verification" in result.stdout
    assert "--part" in result.stdout
    assert "--vitis" in result.stdout


def test_reference_generator_uses_only_the_canonical_public_api() -> None:
    source = (REFERENCE_ROOT / "generate.py").read_text(encoding="utf-8")

    assert "import ravel_hls as ravel" in source
    assert "ravel.convert(" in source
    assert '"Vitis": {"Run": args.vitis}' in source
    assert "RavelConfig" not in source
    assert "convert_from_keras_model" not in source


def test_vanilla_hls4ml_baseline_has_no_generated_source_edit_path() -> None:
    source = (REFERENCE_ROOT / "baseline.py").read_text(encoding="utf-8")

    assert "hls4ml.converters.convert_from_keras_model(" in source
    assert "hls_model.write()" in source
    assert "hls_model.build(" in source
    assert "source_policy\": \"no generated source edits" in source
    for forbidden in ("write_text(", "write_bytes(", "subprocess", "ravel_hls"):
        assert forbidden not in source


def test_vanilla_hls4ml_baseline_exposes_a_lightweight_help_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, str(REFERENCE_ROOT / "baseline.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "vanilla hls4ml CNN-for-Arianna baseline" in result.stdout
    assert "--vitis" in result.stdout


def test_reference_documents_the_reproducible_vanilla_baseline() -> None:
    readme = (REFERENCE_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python references/cnn_for_arianna/baseline.py --vitis" in readme
    assert "6e16cd474bcf45e41b173734b59e70ddd6ed6323" in readme
    assert "65021d84030d9c09a7f1fd541221b150dad14858ad85458912a1a6a6b40a9978" in readme
    assert "no generated C++, headers, Tcl, or YAML are edited" in readme


def test_vitis_2023_2_launcher_adapter_only_translates_the_command(
    tmp_path: Path,
) -> None:
    fake_vitis_hls = tmp_path / "vitis_hls"
    fake_vitis_hls.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_vitis_hls.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment['PATH']}"

    result = subprocess.run(
        [
            str(REFERENCE_ROOT / "tools" / "vitis-run"),
            "--tcl",
            "build_prj.tcl",
            "--mode",
            "hls",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["-f", "build_prj.tcl"]
