import ast
import inspect
from pathlib import Path


def test_v15_renderer_consumes_only_resolved_design_and_parameter_payload() -> None:
    from ravel_hls.rendering.vitis.renderer import render_aria_project

    assert list(inspect.signature(render_aria_project).parameters) == [
        "project_path",
        "project_name",
        "resolved_design",
        "parameter_payload",
    ]
    source_path = Path(inspect.getsourcefile(render_aria_project))
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {"analysis", "extraction", "frontends", "hls4ml"}
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        (node.module or "").lstrip(".").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert imported_roots.isdisjoint(forbidden_modules)
    assert ".get_layers(" not in source
    assert ".get_attr(" not in source
