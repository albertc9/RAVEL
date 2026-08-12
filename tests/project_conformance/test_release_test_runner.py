from pathlib import Path
import subprocess
import sys


def test_release_test_runner_discovers_every_test_module() -> None:
    repository = Path(__file__).resolve().parents[2]
    runner = repository / ".github/scripts/run_release_tests.py"
    completed = subprocess.run(
        [sys.executable, str(runner), "--list"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    expected = sorted(
        path.relative_to(repository).as_posix()
        for path in (repository / "tests").rglob("test_*.py")
        if path.is_file()
    )
    assert completed.stdout.splitlines() == expected
