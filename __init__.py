"""Compatibility wrapper for the mmcv Python package.

This repository's source tree uses a nested structure where the actual package
lives under ``mmcv/mmcv``. When the tree is added to ``PYTHONPATH`` alongside a
binary-only wheel, Python treats ``mmcv`` as a namespace package without
executing the expected ``mmcv.__init__``. As a result attributes such as
``mmcv.__version__`` may be missing, which breaks downstream libraries like
MMDetection.  Installing this shim ensures the canonical package ``mmcv.mmcv``
remains importable while the top-level ``mmcv`` module exposes the same public
API and version metadata.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_mmcv_pkg: ModuleType = import_module("mmcv.mmcv")

# Mirror the public API exported by mmcv.mmcv.
__all__ = getattr(_mmcv_pkg, "__all__", [])
for _name in __all__:
    globals()[_name] = getattr(_mmcv_pkg, _name)

# Expose version helpers expected by downstream projects.
__version__ = getattr(_mmcv_pkg, "__version__", None)
version_info = getattr(_mmcv_pkg, "version_info", None)
parse_version_info: Any | None = getattr(_mmcv_pkg, "parse_version_info", None)

# Delegate attribute access to the real package.
def __getattr__(name: str) -> Any:
    return getattr(_mmcv_pkg, name)

# Ensure package submodules (e.g. mmcv.ops) resolve correctly.
__path__ = _mmcv_pkg.__path__
