"""Protocol definition for residual adapters.

This module defines the common interface that all residual adapters
must implement, allowing for pluggable normalization strategies.
"""

from typing import Protocol

import torch


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
