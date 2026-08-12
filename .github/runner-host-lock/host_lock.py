#!/usr/bin/env python3
"""Serialize jobs from multiple GitHub Actions runners on one small host."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


DEFAULT_ROOT = Path("/home/pipeline/actions-runner-host-lock")
RUNNER_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _log(message: str) -> None:
    print(f"runner-host-lock: {message}", file=sys.stderr, flush=True)


def _positive_number(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise SystemExit(f"{name} must be a number, got {raw!r}") from error
    if value <= 0:
        raise SystemExit(f"{name} must be greater than zero, got {raw!r}")
    return value


def _runner_id() -> str:
    value = os.environ.get("HOST_LOCK_RUNNER_ID", "")
    if not RUNNER_ID_PATTERN.fullmatch(value):
        raise SystemExit(
            "HOST_LOCK_RUNNER_ID must contain only letters, digits, dot, dash, "
            "or underscore"
        )
    return value


def _lock_root() -> Path:
    root = Path(os.environ.get("HOST_LOCK_ROOT", str(DEFAULT_ROOT))).resolve()
    if root == Path(root.anchor):
        raise SystemExit("HOST_LOCK_ROOT cannot be a filesystem root")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _boot_id() -> str:
    override = os.environ.get("HOST_LOCK_BOOT_ID")
    if override:
        return override
    linux_boot_id = Path("/proc/sys/kernel/random/boot_id")
    if linux_boot_id.is_file():
        return linux_boot_id.read_text(encoding="utf-8").strip()
    result = subprocess.run(
        ["sysctl", "-n", "kern.boottime"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    raise SystemExit("cannot determine host boot identity")


def _process_identity(pid: int) -> str | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass

    override = os.environ.get("HOST_LOCK_PROCESS_IDENTITY")
    if override:
        return f"test-override:{override}"

    linux_stat = Path(f"/proc/{pid}/stat")
    if linux_stat.is_file():
        stat = linux_stat.read_text(encoding="utf-8")
        closing_parenthesis = stat.rfind(")")
        fields_after_name = stat[closing_parenthesis + 2 :].split()
        if closing_parenthesis < 0 or len(fields_after_name) <= 19:
            return None
        return f"linux-start-ticks:{fields_after_name[19]}"

    result = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    started = " ".join(result.stdout.split())
    if result.returncode != 0 or not started:
        return None
    return f"ps-lstart:{started}"


@contextmanager
def _control_lock(root: Path):
    with (root / "control.lock").open("a+", encoding="utf-8") as control:
        fcntl.flock(control.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(control.fileno(), fcntl.LOCK_UN)


def _read_owner(active: Path) -> dict[str, object] | None:
    try:
        value = json.loads((active / "owner.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _owner_is_alive(owner: dict[str, object] | None, boot_id: str) -> bool:
    if owner is None or owner.get("boot_id") != boot_id:
        return False
    pid = owner.get("pid")
    identity = owner.get("process_identity")
    if not isinstance(pid, int) or not isinstance(identity, str):
        return False
    return _process_identity(pid) == identity


def _remove_active(active: Path) -> None:
    owner_file = active / "owner.json"
    try:
        owner_file.unlink()
    except FileNotFoundError:
        pass
    active.rmdir()


def _try_acquire(root: Path, runner_id: str, boot_id: str) -> bool:
    active = root / "active"
    state = root / f"{runner_id}.token"
    with _control_lock(root):
        if active.exists():
            owner = _read_owner(active)
            if _owner_is_alive(owner, boot_id):
                return False
            stale_runner = owner.get("runner") if owner else "unknown"
            _log(f"reclaiming stale host lock from {stale_runner}")
            _remove_active(active)

        parent_pid = os.getppid()
        parent_identity = _process_identity(parent_pid)
        if parent_identity is None:
            raise SystemExit(f"cannot identify hook parent process {parent_pid}")

        token = uuid.uuid4().hex
        active.mkdir()
        owner = {
            "token": token,
            "runner": runner_id,
            "pid": parent_pid,
            "process_identity": parent_identity,
            "boot_id": boot_id,
            "acquired_epoch": int(time.time()),
        }
        (active / "owner.json").write_text(
            json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
        )
        state.write_text(token + "\n", encoding="utf-8")
        return True


def acquire() -> int:
    runner_id = _runner_id()
    root = _lock_root()
    boot_id = _boot_id()
    poll_seconds = _positive_number("HOST_LOCK_POLL_SECONDS", 2)
    timeout_seconds = _positive_number("HOST_LOCK_TIMEOUT_SECONDS", 21600)
    deadline = time.monotonic() + timeout_seconds
    last_wait_log = 0.0

    while True:
        if _try_acquire(root, runner_id, boot_id):
            _log(f"acquired by {runner_id}")
            return 0
        now = time.monotonic()
        if now >= deadline:
            _log(f"timed out after {timeout_seconds:g}s waiting for host lock")
            return 1
        if now - last_wait_log >= 60 or last_wait_log == 0:
            owner = _read_owner(root / "active") or {}
            _log(f"waiting for {owner.get('runner', 'another runner')}")
            last_wait_log = now
        time.sleep(min(poll_seconds, max(0.0, deadline - now)))


def release() -> int:
    runner_id = _runner_id()
    root = _lock_root()
    state = root / f"{runner_id}.token"
    try:
        token = state.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        _log(f"no lock token recorded for {runner_id}; leaving active lock unchanged")
        return 0

    with _control_lock(root):
        active = root / "active"
        owner = _read_owner(active)
        if owner is None or owner.get("token") != token:
            _log(f"{runner_id} does not own the active lock; leaving it unchanged")
            state.unlink(missing_ok=True)
            return 0
        _remove_active(active)
        state.unlink(missing_ok=True)

    _log(f"released by {runner_id}")
    return 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"acquire", "release"}:
        print(f"usage: {Path(sys.argv[0]).name} acquire|release", file=sys.stderr)
        return 2
    return acquire() if sys.argv[1] == "acquire" else release()


if __name__ == "__main__":
    raise SystemExit(main())
