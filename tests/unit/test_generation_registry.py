from dataclasses import FrozenInstanceError

import pytest

from ravel_hls.generations import builtin_generation


def test_aria_generation_declares_its_complete_builtin_extension_boundary() -> None:
    generation = builtin_generation("aria", "1.5.0")

    assert generation.identity == {"id": "aria", "version": "1.5.0"}
    assert [component.id for component in generation.operation_extractors] == [
        "input",
        "repack",
        "conv2d",
        "relu",
        "max_pool2d",
        "reshape",
        "dense",
    ]
    assert [matcher.id for matcher in generation.family_matchers] == [
        "hgq-conv-pool-dense"
    ]
    assert [strategy.id for strategy in generation.strategies] == [
        "aria-wide-stream"
    ]
    assert generation.resolver.id == "aria-explicit-pd"
    assert [item.id for item in generation.passes] == [
        "pack-temporal-input",
        "fuse-repack-into-first-conv",
        "propagate-wide-relu-stream",
        "specialize-nonoverlapping-maxpool",
        "stream-flatten-into-dense",
        "bind-shallow-internal-fifos",
        "elide-dataflow-start-propagation",
    ]
    assert [(item.backend, item.io_type) for item in generation.backends] == [
        ("Vitis", "io_stream")
    ]


def test_builtin_generation_registry_is_immutable_and_closed() -> None:
    generation = builtin_generation("aria", "1.5.0")

    with pytest.raises(FrozenInstanceError):
        generation.version = "next"
    with pytest.raises(LookupError, match="unknown RAVEL generation"):
        builtin_generation("aria", "9.9.9")
