from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
ACQUIRE = ROOT / ".github" / "runner-host-lock" / "acquire.sh"
RELEASE = ROOT / ".github" / "runner-host-lock" / "release.sh"


def _environment(lock_root: Path, runner_id: str) -> dict[str, str]:
    return {
        **os.environ,
        "HOST_LOCK_ROOT": str(lock_root),
        "HOST_LOCK_RUNNER_ID": runner_id,
        "HOST_LOCK_POLL_SECONDS": "1",
        "HOST_LOCK_TIMEOUT_SECONDS": "10",
        "HOST_LOCK_BOOT_ID": "runner-host-lock-test-boot",
        "HOST_LOCK_PROCESS_IDENTITY": "stable-test-parent",
    }


def _wait_for(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def _runner_process(
    lock_root: Path,
    runner_id: str,
    acquired_marker: Path,
    release_marker: Path,
) -> subprocess.Popen[str]:
    command = """
set -eu
"$1"
: > "$2"
while [ ! -e "$3" ]; do
    sleep 0.05
done
"$4"
"""
    return subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            command,
            "runner-host-lock-test",
            str(ACQUIRE),
            str(acquired_marker),
            str(release_marker),
            str(RELEASE),
        ],
        env=_environment(lock_root, runner_id),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_two_runner_jobs_are_serialized(tmp_path: Path) -> None:
    lock_root = tmp_path / "host-lock"
    first_acquired = tmp_path / "first-acquired"
    first_release = tmp_path / "first-release"
    second_acquired = tmp_path / "second-acquired"
    second_release = tmp_path / "second-release"

    first = _runner_process(lock_root, "panel-runner", first_acquired, first_release)
    _wait_for(first_acquired)
    second = _runner_process(lock_root, "ravel-runner", second_acquired, second_release)

    time.sleep(1.2)
    assert not second_acquired.exists()

    first_release.touch()
    _wait_for(second_acquired)
    second_release.touch()

    assert first.communicate(timeout=5)[0] == ""
    assert first.returncode == 0
    assert second.communicate(timeout=5)[0] == ""
    assert second.returncode == 0
    assert not (lock_root / "active").exists()


def test_dead_owner_lock_is_recovered(tmp_path: Path) -> None:
    lock_root = tmp_path / "host-lock"
    active = lock_root / "active"
    active.mkdir(parents=True)
    (active / "owner.json").write_text(
        json.dumps(
            {
                "token": "dead-owner-token",
                "runner": "dead-runner",
                "pid": 99999999,
                "process_identity": "linux-start-ticks:0",
                "boot_id": "runner-host-lock-test-boot",
                "acquired_epoch": 0,
            }
        )
    )

    acquired = subprocess.run(
        [str(ACQUIRE)],
        env=_environment(lock_root, "replacement-runner"),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "reclaiming stale host lock" in acquired.stderr

    subprocess.run(
        [str(RELEASE)],
        env=_environment(lock_root, "replacement-runner"),
        check=True,
    )
    assert not active.exists()


def test_non_owner_cannot_release_lock(tmp_path: Path) -> None:
    lock_root = tmp_path / "host-lock"
    subprocess.run(
        [str(ACQUIRE)],
        env=_environment(lock_root, "owner-runner"),
        check=True,
    )

    subprocess.run(
        [str(RELEASE)],
        env=_environment(lock_root, "other-runner"),
        check=True,
    )
    assert (lock_root / "active").is_dir()

    subprocess.run(
        [str(RELEASE)],
        env=_environment(lock_root, "owner-runner"),
        check=True,
    )
    assert not (lock_root / "active").exists()
