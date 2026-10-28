from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    if name in {"app", "create_app"}:
        main_module = import_module("app.main")
        return getattr(main_module, name)
    raise AttributeError(f"module 'app' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
