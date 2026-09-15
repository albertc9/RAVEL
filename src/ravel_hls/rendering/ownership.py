"""Track declared source overlays against the untouched generated baseline."""

from dataclasses import dataclass
from pathlib import Path

from ..exceptions import ProjectGenerationError
from ..manifest import _build_source_closure


@dataclass(frozen=True)
class SourceStep:
    id: str
    version: int
    paths: tuple[str, ...]


@dataclass(frozen=True)
class OwnedChange:
    path: str
    owner: str
    version: int
    before_sha256: str | None
    after_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "owner": {"id": self.owner, "version": self.version},
                "before_sha256": self.before_sha256, "after_sha256": self.after_sha256}


class SourceOwnership:
    """Transaction-local source ledger; unrecorded modifications fail closed."""

    def __init__(self, root: Path):
        self.root = root
        self._expected = self._snapshot()
        self._changes: list[OwnedChange] = []

    def _snapshot(self) -> dict[str, str]:
        return {entry["path"]: entry["sha256"] for entry in _build_source_closure(self.root)}

    def record(self, owner: str, version: int, paths: list[str]) -> None:
        self.record_steps((SourceStep(owner, version, tuple(paths)),))

    def record_steps(self, steps: tuple[SourceStep, ...]) -> None:
        owners = {}
        for step in steps:
            for path in step.paths:
                if path in owners:
                    raise ProjectGenerationError(f"Source composition has ambiguous ownership: {path}")
                owners[path] = step
        observed = self._snapshot()
        changed = {path for path in observed.keys() | self._expected.keys()
                   if observed.get(path) != self._expected.get(path)}
        undeclared = changed - owners.keys()
        deleted = self._expected.keys() - observed.keys()
        if undeclared or deleted:
            raise ProjectGenerationError("Source composition contains unrecorded mutations: " + ", ".join(sorted(undeclared | deleted)))
        self._changes.extend(OwnedChange(path, owners[path].id, owners[path].version, self._expected.get(path), observed[path]) for path in sorted(changed))
        self._expected = observed

    def verify(self) -> list[dict[str, object]]:
        if self._snapshot() != self._expected:
            raise ProjectGenerationError("Source composition contains an unrecorded mutation")
        return [change.to_dict() for change in self._changes]
