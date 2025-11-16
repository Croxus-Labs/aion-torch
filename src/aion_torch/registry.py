"""Registry and factory system for residual adapters.

This module provides a simple registration mechanism to enable
pluggable adapter implementations without modifying core code.
"""

from collections.abc import Callable
from typing import Any

# Internal registry storage
_registry: dict[str, Callable] = {}


def register_adapter(name: str, ctor: Callable) -> None:
    """Register a residual adapter constructor.

    Args:
        name: Unique identifier for this adapter
        ctor: Constructor callable that returns an adapter instance

    Example:
        >>> from aion_torch import AionResidual, register_adapter
        >>> register_adapter("aion", AionResidual)
        >>> register_adapter("custom", MyCustomAdapter)
    """
    _registry[name] = ctor


def make_adapter(name: str, **kwargs: Any):
    """Create an adapter instance from the registry.

    Args:
        name: Name of registered adapter
        **kwargs: Arguments to pass to adapter constructor

    Returns:
        Adapter instance

    Raises:
        KeyError: If adapter name is not registered

    Example:
        >>> adapter = make_adapter("aion", alpha0=0.1, beta=0.05)
    """
    if name not in _registry:
        raise KeyError(f"Adapter '{name}' not found. Available: {list(_registry.keys())}")
    return _registry[name](**kwargs)


def list_adapters() -> list:
    """List all registered adapter names.

    Returns:
        List of registered adapter names
    """
    return list(_registry.keys())
