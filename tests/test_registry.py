"""Tests for registry system."""

import pytest
import torch
import torch.nn as nn

from aion_torch.registry import (  # type: ignore
    _registry,
    list_adapters,
    make_adapter,
    register_adapter,
)


class DummyAdapter(nn.Module):
    """Dummy adapter for testing."""

    def __init__(self, param1: float = 1.0, param2: int = 2) -> None:
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dummy forward pass."""
        return x


class TestRegistry:
    """Test suite for registry system."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        _registry.clear()

    def test_register_adapter(self) -> None:
        """Test registering an adapter."""
        register_adapter("dummy", DummyAdapter)

        assert "dummy" in _registry
        assert _registry["dummy"] is DummyAdapter

    def test_make_adapter(self) -> None:
        """Test creating adapter from registry."""
        register_adapter("dummy", DummyAdapter)

        adapter = make_adapter("dummy", param1=5.0, param2=10)

        assert isinstance(adapter, DummyAdapter)
        assert adapter.param1 == 5.0
        assert adapter.param2 == 10

    def test_make_adapter_default_params(self) -> None:
        """Test creating adapter with default parameters."""
        register_adapter("dummy", DummyAdapter)

        adapter = make_adapter("dummy")

        assert isinstance(adapter, DummyAdapter)
        assert adapter.param1 == 1.0
        assert adapter.param2 == 2

    def test_make_adapter_not_registered(self) -> None:
        """Test error when adapter is not registered."""
        with pytest.raises(KeyError) as exc_info:
            make_adapter("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_list_adapters_empty(self) -> None:
        """Test listing adapters when registry is empty."""
        adapters = list_adapters()

        assert isinstance(adapters, list)
        assert len(adapters) == 0

    def test_list_adapters(self) -> None:
        """Test listing registered adapters."""
        register_adapter("dummy1", DummyAdapter)
        register_adapter("dummy2", DummyAdapter)
        register_adapter("dummy3", DummyAdapter)

        adapters = list_adapters()

        assert len(adapters) == 3
        assert "dummy1" in adapters
        assert "dummy2" in adapters
        assert "dummy3" in adapters

    def test_register_overwrite(self) -> None:
        """Test that registering same name overwrites previous."""
        register_adapter("dummy", DummyAdapter)

        class AnotherAdapter(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        register_adapter("dummy", AnotherAdapter)

        adapter = make_adapter("dummy")
        assert isinstance(adapter, AnotherAdapter)

    def test_register_lambda(self) -> None:
        """Test registering a lambda constructor."""
        register_adapter("lambda_adapter", lambda x: {"value": x})

        adapter = make_adapter("lambda_adapter", x=42)

        assert adapter == {"value": 42}

    def test_multiple_instances(self) -> None:
        """Test creating multiple instances from same registration."""
        register_adapter("dummy", DummyAdapter)

        adapter1 = make_adapter("dummy", param1=1.0)
        adapter2 = make_adapter("dummy", param1=2.0)

        assert adapter1.param1 == 1.0
        assert adapter2.param1 == 2.0
        assert adapter1 is not adapter2
