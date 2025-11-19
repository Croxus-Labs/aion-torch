"""Protocol definition and convenience blocks for residual adapters.

This module defines the common interface that all residual adapters
must implement, and provides high-level blocks for common patterns.
"""

from typing import Protocol

import torch
import torch.nn as nn

from .aion_adapter import AionResidual


class ResidualAdapter(Protocol):
    """Protocol for residual connection adapters.

    Any residual adapter (AION, PreLN, etc.) should implement
    this interface to be compatible with the registry system.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply adaptive residual connection.

        Args:
            x: Input tensor (residual branch input)
            y: Transform output tensor (e.g., from FFN, attention)

        Returns:
            Combined output, typically: x + scale(y)

        Notes:
            - x and y must have the same shape
            - scale() can be fixed or adaptive (AION)
        """


class AionBlock(nn.Module):
    """AION-stabilized residual block implementing Pre-LayerNorm pattern.

    Structure:
        x_norm = Norm(x)
        y = Layer(x_norm)
        output = AionResidual(x, y)

    This block encapsulates the standard Pre-LayerNorm architecture improved
    with AION adaptive scaling, ensuring the exact pattern assumed by the
    theoretical guarantees.

    Args:
        layer: The neural network layer (Attention, MLP, etc.)
        dim: Dimension of input features (for normalization)
        alpha0: Initial scaling factor for AION (default: 0.1)
        beta: Adaptation strength for AION (default: 0.05)
        ema_gamma: EMA smoothing factor (default: 0.99)
        epsilon: Numerical stability constant (default: 1e-8)
    """

    def __init__(
        self,
        layer: nn.Module,
        dim: int,
        alpha0: float = 0.1,
        beta: float = 0.05,
        ema_gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer = layer
        self.aion = AionResidual(
            alpha0=alpha0, beta=beta, ema_gamma=ema_gamma, epsilon=epsilon
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the AION block transformation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with adaptive residual connection
        """
        x_norm = self.norm(x)
        y = self.layer(x_norm)
        return self.aion(x, y)  # type: ignore
