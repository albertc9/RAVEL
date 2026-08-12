"""The closed, explicit Aria 1.5 built-in generation definition."""

from ...rendering.vitis import render_aria_project
from ..registry import (
    BackendBindingDefinition,
    ComponentDefinition,
    FamilyMatcherDefinition,
    GenerationDefinition,
    ResolverDefinition,
    StrategyDefinition,
)
from .matching import evaluate_aria_wide_stream, match_hgq_conv_pool_dense
from .passes import ARIA_PASS_DEFINITIONS, resolve_aria_design


ARIA_1_5_1 = GenerationDefinition(
    id="aria",
    version="1.5.1",
    operation_extractors=tuple(
        ComponentDefinition(operation_id, 1)
        for operation_id in (
            "input",
            "repack",
            "conv2d",
            "relu",
            "max_pool2d",
            "reshape",
            "dense",
        )
    ),
    family_matchers=(
        FamilyMatcherDefinition(
            "hgq-conv-pool-dense", 1, match_hgq_conv_pool_dense
        ),
    ),
    strategies=(
        StrategyDefinition("aria-wide-stream", 2, evaluate_aria_wide_stream),
    ),
    resolver=ResolverDefinition("aria-explicit-pd", 2, resolve_aria_design),
    passes=ARIA_PASS_DEFINITIONS,
    backends=(
        BackendBindingDefinition(
            "Vitis", "io_stream", "aria-vitis-templates", 2, render_aria_project
        ),
    ),
)
