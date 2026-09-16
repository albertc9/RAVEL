"""Bind selected render steps to the function identities emitted by Vitis."""

import re
from collections.abc import Mapping
from typing import Any


def report_bindings(design: Mapping[str, Any]) -> list[dict[str, Any]]:
    rendering = design["rendering"]
    native = rendering["native_operations"]
    bindings = []
    for stage in design["stages"]:
        strategy = stage["strategy"]["id"]
        operation_ids = stage["operation_ids"]
        if strategy == "identity-layout-view":
            bindings.append({"stage_id": operation_ids[0], "functions": [], "realization": "zero-hardware-view"})
            continue
        if strategy == "aria-window-stream":
            functions = [{"name": "scheduled_conv", "config": native[operation_ids[0]]["config_symbol"]},
                         {"name": "relu", "config": native[operation_ids[1]]["config_symbol"].replace("config", "relu_config")},
                         {"name": "scheduled_pool", "config": native[operation_ids[2]]["config_symbol"]}]
        elif strategy == "hls4ml-temporal-block":
            functions = []
            for operation in operation_ids:
                call = native[operation]["native_call"]
                functions.append({"name": re.match(r"nnet::(\w+)", call)[1],
                                  "config": native[operation]["config_symbol"].replace("config", "relu_config") if operation.startswith("relu_") else native[operation]["config_symbol"]})
        elif strategy == "phara":
            functions = [{"name": rendering["phara_fused_function"]}]
        elif strategy == "aria-wide-stream":
            functions = [{"name": rendering["first_convolution_function"]},
                         {"name": "maxpool2d_wide_nonoverlap_cl"}]
        else:
            functions = [{"name": "dense_wide_stream"}]
        bindings.append({"stage_id": operation_ids[0], "functions": functions})
    for index, bridge in enumerate(design["bridges"]):
        from math import prod
        bindings.append({"stage_id": f"bridge_{index}", "functions": [{"name": "repack", "config": f"{prod(bridge['input']['shape'])}u"}]})
    return bindings
