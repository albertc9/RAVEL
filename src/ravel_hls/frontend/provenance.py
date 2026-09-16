"""Qualified static HGQ source contracts, independent of parameter values."""
from collections.abc import Mapping
from typing import Any
from .extraction import _json_value
_QUANTIZER_ROLES = {
    "iq_conf": "input",
    "kq_conf": "weight",
    "bq_conf": "bias",
    "oq_conf": "output",
}


def _frontend_provenance(model: Any) -> dict[str, Any]:
    source_layers = []
    quantizer_contracts = []
    for ordinal, layer in enumerate(model.layers):
        source_layers.append(
            {
                "ordinal": ordinal,
                "module": type(layer).__module__,
                "class_name": type(layer).__name__,
            }
        )
        config = layer.get_config()
        for field, role in _QUANTIZER_ROLES.items():
            serialized = config.get(field)
            if not isinstance(serialized, Mapping):
                continue
            quantizer = serialized.get("config")
            if not isinstance(quantizer, Mapping):
                continue
            quantizer_contracts.append(
                {
                    "source_ordinal": ordinal,
                    "role": role,
                    "q_type": quantizer.get("q_type"),
                    "rounding": quantizer.get("round_mode"),
                    "overflow": quantizer.get("overflow_mode"),
                    "homogeneous_axis": _json_value(
                        quantizer.get("homogeneous_axis")
                    ),
                    "heterogeneous_axis": _json_value(
                        quantizer.get("heterogeneous_axis")
                    ),
                    "is_weight": quantizer.get("is_weight"),
                }
            )
    return {
        "adapter": {"id": "keras-hgq2", "version": 1},
        "source_layers": source_layers,
        "quantizer_contracts": quantizer_contracts,
    }


