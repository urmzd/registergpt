"""Auto-discovery registry for AgiModel subclasses."""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from core.base import AgiModel

# Model modules live in v*/ directories at the repo root.
_MODEL_PACKAGES = [
    p.name + ".model"
    for p in sorted(Path(__file__).resolve().parent.parent.iterdir())
    if p.is_dir() and p.name.startswith("v") and (p / "model.py").exists()
]

_REGISTRY: dict[str, type[AgiModel]] | None = None


def _discover() -> dict[str, type[AgiModel]]:
    """Import all model modules and collect AgiModel subclasses with a version set."""
    registry: dict[str, type[AgiModel]] = {}
    for module_name in _MODEL_PACKAGES:
        mod = importlib.import_module(module_name)
        for attr in dir(mod):
            cls = getattr(mod, attr)
            if (
                isinstance(cls, type)
                and issubclass(cls, AgiModel)
                and cls is not AgiModel
                and cls.version
            ):
                registry[cls.version] = cls
    return registry


def get_registry() -> dict[str, type[AgiModel]]:
    """Return the version -> class mapping, discovering on first call."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _discover()
    return _REGISTRY


def build_model(version: str, args) -> AgiModel:
    """Instantiate a model by version string."""
    registry = get_registry()
    if version not in registry:
        raise ValueError(
            f"Unknown model version: {version!r}. "
            f"Available: {sorted(registry.keys())}"
        )
    cls = registry[version]
    return cls(**cls.build_kwargs(args))
