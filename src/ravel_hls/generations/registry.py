"""Immutable internal extension definitions for built-in generations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..domain import ParameterPayload


@dataclass(frozen=True)
class ComponentDefinition:
    id: str
    version: int


@dataclass(frozen=True)
class FamilyMatcherDefinition(ComponentDefinition):
    evaluator: Callable[..., tuple[dict[str, Any] | None, dict[str, Any]]] = field(
        repr=False, compare=False
    )

    def evaluate(
        self, facts: Mapping[str, Any], provenance: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return self.evaluator(facts, provenance)


@dataclass(frozen=True)
class StrategyDefinition(ComponentDefinition):
    evaluator: Callable[..., list[dict[str, Any]]] = field(
        repr=False, compare=False
    )

    def evaluate(
        self,
        operations: list[Mapping[str, Any]],
        choices: Mapping[str, int],
        plan: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        return self.evaluator(operations, choices, plan)


@dataclass(frozen=True)
class ResolverDefinition(ComponentDefinition):
    evaluator: Callable[..., dict[str, Any]] = field(repr=False, compare=False)

    def resolve(self, **values: Any) -> dict[str, Any]:
        return self.evaluator(**values)


@dataclass(frozen=True)
class BackendBindingDefinition:
    backend: str
    io_type: str
    renderer_id: str
    renderer_version: int
    renderer: Callable[
        [Path, str, Mapping[str, Any], ParameterPayload], list[str]
    ] = field(repr=False, compare=False)

    def render(
        self,
        project_path: Path,
        project_name: str,
        resolved_design: Mapping[str, Any],
        parameter_payload: ParameterPayload,
    ) -> list[str]:
        return self.renderer(
            project_path, project_name, resolved_design, parameter_payload
        )


@dataclass(frozen=True)
class GenerationDefinition:
    id: str
    version: str
    operation_extractors: tuple[ComponentDefinition, ...]
    family_matchers: tuple[FamilyMatcherDefinition, ...]
    strategies: tuple[StrategyDefinition, ...]
    resolver: ResolverDefinition
    passes: tuple[ComponentDefinition, ...]
    backends: tuple[BackendBindingDefinition, ...]

    @property
    def identity(self) -> dict[str, str]:
        return {"id": self.id, "version": self.version}

    def match_model_family(
        self, facts: Mapping[str, Any], provenance: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        results = [matcher.evaluate(facts, provenance) for matcher in self.family_matchers]
        matches = [result for result in results if result[0] is not None]
        if len(matches) > 1:
            identities = [result[0] for result in matches]
            raise RuntimeError(f"Ambiguous model-family match: {identities}")
        if matches:
            return matches[0]
        findings = [
            finding
            for _, applicability in results
            for finding in applicability["findings"]
        ]
        return None, {"status": "unsupported", "findings": findings}

    def strategy(self, strategy_id: str, version: int) -> StrategyDefinition:
        for strategy in self.strategies:
            if (strategy.id, strategy.version) == (strategy_id, version):
                return strategy
        raise LookupError(
            f"unknown strategy for {self.id} {self.version}: "
            f"{strategy_id} v{version}"
        )

    def backend_binding(
        self, backend: str, io_type: str
    ) -> BackendBindingDefinition:
        for binding in self.backends:
            if (binding.backend, binding.io_type) == (backend, io_type):
                return binding
        raise LookupError(
            f"unknown backend binding for {self.id} {self.version}: "
            f"{backend}/{io_type}"
        )
