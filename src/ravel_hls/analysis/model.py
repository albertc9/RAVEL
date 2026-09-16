"""Thin public orchestration over conversion, projection, capability and planning."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from ..analysis.dense import analyze_dense_facts
from ..compatibility.dependencies import inspect_dependencies
from ..config import validate_public_config
from ..domain import ParameterPayload
from ..domain.graph import GraphFacts
from ..exceptions import CompatibilityError, ConfigurationError
from ..frontend.hls4ml import convert_model
from ..frontend.extraction import _extract_model_facts, _extract_parameter_payload, _native_rendering_contract, ordered_layers
from ..generations import builtin_generation
from ..identity import ARIA_ID, ARIA_VERSION
from ..manifest import canonical_sha256
from ..planning.design import resolve_model_design

def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value

def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)

@dataclass(frozen=True)
class ModelAnalysis:
    """Stable read-only result returned by :func:`analyze`."""

    _report: Mapping[str, Any]

    @classmethod
    def _from_report(cls, report: Mapping[str, Any]) -> "ModelAnalysis":
        return cls(_freeze(report))

    @property
    def applicable(self) -> bool:
        """Whether one model family and implementation design are applicable."""

        return self._report["applicability"]["status"] == "applicable"

    @property
    def model_family(self) -> Mapping[str, Any] | None:
        """The uniquely matched model family, if any."""

        return self._report["model_family"]

    @property
    def findings(self) -> tuple[Mapping[str, Any], ...]:
        """Structured applicability findings."""

        return self._report["applicability"]["findings"]

    @property
    def model_facts(self) -> Mapping[str, Any]:
        """Immutable normalized hardware-semantic model facts."""

        return self._report["model_facts"]

    @property
    def resolved_design(self) -> Mapping[str, Any] | None:
        """The immutable render-ready design when analysis is applicable."""

        return self._report["resolved_design"]

    def to_dict(self) -> dict[str, Any]:
        """Return an independent schema representation."""

        return _thaw(self._report)

@dataclass(frozen=True)
class _AnalyzedModel:
    graph: Any
    source_model: Any
    analysis: ModelAnalysis
    parameter_payload: ParameterPayload

def analyze(model: Any, config: Mapping[str, Any]) -> ModelAnalysis:
    """Analyze a qualified Keras/HGQ2 model without publishing a project."""

    return _analyze_model(model, config).analysis

def _analyze_model(model: Any, config: Mapping[str, Any], *, recorded_manifest=None) -> _AnalyzedModel:
    """Return the private graph-bearing analysis used by conversion."""

    generation = builtin_generation(ARIA_ID, ARIA_VERSION)
    config = validate_public_config(config)
    hls_values = config["HLS"]
    backend = hls_values.get("Backend", "Vitis")
    io_type = hls_values.get("IOType", "io_stream")
    if backend != "Vitis":
        raise ConfigurationError("HLS.Backend must be Vitis")
    if io_type != "io_stream":
        raise ConfigurationError("HLS.IOType must be io_stream")

    choices = config["Optimization"]

    dependencies = inspect_dependencies()
    if dependencies["dependency_qualification"] != "qualified":
        failed = [
            name
            for name, facts in dependencies["dependencies"].items()
            if facts["status"] != "qualified"
        ]
        raise CompatibilityError(
            f"Aria {ARIA_VERSION} dependency stack is not qualified: " + ", ".join(failed)
        )

    graph, normalized_model, frontend_provenance = convert_model(model, hls_values)
    layers = ordered_layers(graph)
    by_name = {layer.name: layer for layer in layers}
    projected, fingerprints = _extract_model_facts(layers, input_ports=tuple(by_name[name].outputs[0] for name in graph.inputs), output_ports=tuple(graph.outputs))
    typed_facts = GraphFacts.from_dict(projected)
    model_facts = typed_facts.to_dict()
    parameter_payload = _extract_parameter_payload(layers)
    fingerprints["frontend_provenance_sha256"] = canonical_sha256(frontend_provenance)
    native = _native_rendering_contract(layers)
    dense_facts = {"dense": analyze_dense_facts(layers)}
    recorded_design = None
    generation_identity = generation.identity
    if recorded_manifest is not None:
        if fingerprints["model_structure_sha256"] != recorded_manifest["source_model"]["fingerprints"]["model_structure_sha256"]:
            raise CompatibilityError("Refresh model changes the recorded architecture contract; use ordinary conversion")
        recorded_design = recorded_manifest["resolved_design"]
        if canonical_sha256(recorded_design) != recorded_manifest["generated_plan_sha256"]:
            raise CompatibilityError("Recorded plan identity does not match its manifest")
        generation_identity = recorded_manifest["profile"]["generation"]
    model_family, applicability, resolved_design, multi_report = resolve_model_design(
        generation, typed_facts, frontend_provenance, choices, parameter_payload, native, dense_facts, hls_values, recorded_design)
    if resolved_design is None:
        dense_facts = {} if model_family is None else dense_facts
    analysis = ModelAnalysis._from_report(
        {
            **multi_report,
            "schema_version": 1,
            "generation": generation_identity,
            "model_family": model_family,
            "applicability": applicability,
            "frontend_provenance": frontend_provenance,
            "model_facts": {**model_facts, **dense_facts},
            "resolved_design": resolved_design,
            "fingerprints": fingerprints,
        }
    )
    return _AnalyzedModel(graph, normalized_model, analysis, parameter_payload)
