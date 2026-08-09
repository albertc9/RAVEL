import hashlib
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
