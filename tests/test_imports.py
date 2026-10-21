import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HEAVY_MODULES = ("torch", "diffusers", "openai")


def _imports_without_heavy_modules(module):
    code = (
        f"import sys, {module}\n"
        f"loaded = [m for m in {HEAVY_MODULES!r} if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
    )


def test_importing_backend_is_side_effect_free():
    proc = _imports_without_heavy_modules("backend")
    assert proc.returncode == 0, proc.stderr


def test_importing_gui_is_side_effect_free():
    pytest.importorskip("tkinter")
    proc = _imports_without_heavy_modules("gui.app")
    assert proc.returncode == 0, proc.stderr
