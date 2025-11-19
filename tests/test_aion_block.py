"""Tests for AionBlock."""

import torch
import torch.nn as nn

from aion_torch.adapters import AionBlock


class TestAionBlock:
    """Test suite for AionBlock."""

    def test_initialization(self) -> None:
        """Test block initialization."""
        dim = 128
        layer = nn.Linear(dim, dim)
        block = AionBlock(layer, dim)

        assert isinstance(block.norm, nn.LayerNorm)
        assert block.layer is layer
        assert torch.allclose(block.aion.alpha0, torch.tensor(0.1))

    def test_forward(self) -> None:
        """Test basic forward pass."""
        dim = 32
        layer = nn.Linear(dim, dim)
        block = AionBlock(layer, dim)

        x = torch.randn(4, dim)
        out = block(x)

        assert out.shape == x.shape
        # Output should be different from input (residual + transformation)
        assert not torch.allclose(out, x)

    def test_forward_with_custom_params(self) -> None:
        """Test forward with custom parameters."""
        dim = 32
        layer = nn.Linear(dim, dim)
        block = AionBlock(layer, dim, alpha0=0.5, beta=0.1, ema_gamma=0.9, epsilon=1e-6)

        assert torch.allclose(block.aion.alpha0, torch.tensor(0.5))
        assert torch.allclose(block.aion.beta, torch.tensor(0.1))
        assert block.aion.ema_gamma == 0.9
        assert block.aion.epsilon == 1e-6

        x = torch.randn(4, dim)
        out = block(x)
        assert out.shape == x.shape
