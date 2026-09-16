from dataclasses import FrozenInstanceError

import pytest

from ravel_hls.domain.graph import GraphFacts


def test_graph_facts_preserve_opaque_operations_and_cannot_be_mutated() -> None:
    source = {
        "schema_version": 1,
        "inputs": ["input_0:out0"],
        "outputs": ["custom_0:out0"],
        "operations": [
            {"id": "input_0", "kind": "input", "inputs": [],
             "outputs": [{"id": "input_0:out0", "shape": [9, 3, 2],
                          "numeric_type": NUMERIC}],
             "attributes": {"input_shape": [9, 3, 2]}, "parameters": []},
            {"id": "custom_0", "kind": "custom", "inputs": ["input_0:out0"],
             "outputs": [{"id": "custom_0:out0", "shape": [9, 3, 2],
                          "numeric_type": NUMERIC}],
             "attributes": {"opaque_option": "preserved"}, "parameters": []},
        ],
    }

    facts = GraphFacts.from_dict(source)

    assert facts.to_dict() == source
    assert facts.tensor("custom_0:out0").shape == (9, 3, 2)
    assert facts.operation("custom_0").attribute("opaque_option") == "preserved"
    with pytest.raises(FrozenInstanceError):
        facts.operations[0].kind = "changed"
    source["operations"][0]["outputs"][0]["shape"][0] = 99
    assert facts.tensor("input_0:out0").shape == (9, 3, 2)
    assert hash(facts) == hash(GraphFacts.from_dict(facts.to_dict()))


NUMERIC = {
    "kind": "fixed", "width": 10, "integer": 5, "signed": True,
    "rounding": "RND", "saturation": "SAT_SYM", "saturation_bits": 0,
}


def test_temporal_layout_view_proves_feature_major_channel_minor_flattening():
    from ravel_hls.domain.temporal import FeatureLayout
    layout = FeatureLayout.temporal((3, 2, 4))
    assert layout.scalar_index((1, 1, 2)) == 14
    assert layout.coordinates(14) == (1, 1, 2)
    assert layout.to_dict() == {"order": "C", "axes": [
        {"name": "temporal", "extent": 3, "stride": 8},
        {"name": "feature", "extent": 2, "stride": 4},
        {"name": "channel", "extent": 4, "stride": 1},
    ], "scalar_count": 24}
