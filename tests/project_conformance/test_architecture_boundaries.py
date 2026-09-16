"""The agreed domain/planning/rendering seams cannot load compiler state."""

import os
import subprocess
import sys


def test_pure_planning_and_rendering_import_without_frontend_or_vendor_modules():
    code = '''import importlib
import importlib.abc
import sys
class NoCompilerState(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("keras", "tensorflow", "hls4ml", "ravel_hls.frontend", "ravel_hls.analysis.model")
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise AssertionError("Pure seam attempted to import compiler state: " + fullname)
sys.meta_path.insert(0, NoCompilerState())
for name in ("ravel_hls.domain.graph", "ravel_hls.domain.temporal", "ravel_hls.domain.layout",
             "ravel_hls.planning.chain", "ravel_hls.planning.bridges", "ravel_hls.planning.strategies",
             "ravel_hls.rendering.vitis.composed"):
    importlib.import_module(name)
'''
    result = subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
