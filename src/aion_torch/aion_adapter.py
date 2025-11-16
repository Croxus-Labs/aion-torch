"""AION residual adapter implementation.

This module provides the AION (Adaptive Input/Output Normalization)
residual connection that adaptively scales the residual branch based
on energy ratios between input and output tensors.
"""

import torch
import torch.nn as nn

from .alpha import compute_alpha
from .energy import energy


class AionResidual(nn.Module):
    """AION adaptive residual connection.

    Implements adaptive scaling of residual connections based on energy ratios:
    α = α₀ / (1 + β · ratio_s) where ratio_s = E[y]/(E[x] + ε)

    The ratio_s can be optionally EMA-smoothed for stability.
    """

    def __init__(
        self,
        alpha0: float = 0.1,
        beta: float = 0.05,
        ema_gamma: float = 0.99,
        k_update: int = 1,
        epsilon: float = 1e-8,
    ):
        """Initialize AION residual adapter.

        Args:
            alpha0: Base scaling parameter (learnable)
            beta: Adaptation coefficient (learnable)
            ema_gamma: EMA smoothing factor for ratio_s (1.0 = no smoothing)
            k_update: Update alpha every k steps (for efficiency)
            epsilon: Small constant for numerical stability
        """
        super().__init__()

        # Learnable parameters
        self.alpha0 = nn.Parameter(torch.tensor(alpha0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

        # Configuration
        self.ema_gamma = ema_gamma
        self.k_update = k_update
        self.epsilon = epsilon

        # State
        self.register_buffer("ratio_ema", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.int32))
        self._alpha_cached: torch.Tensor | None = None

        # Type hints for buffers (for static type checkers)
        self.ratio_ema: torch.Tensor
        self.step_count: torch.Tensor

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply adaptive residual connection.

        Args:
            x: Input tensor (residual branch input)
            y: Transform output tensor (e.g., from FFN, attention)

        Returns:
            Combined output: x + α · y where α is adaptive
        """
        if self.training and self.step_count.item() % self.k_update == 0:
            # Compute energies
            ex = energy(x, dim=-1, keepdim=False)  # [B, T] or [B] depending on dims
            ey = energy(y, dim=-1, keepdim=False)

            # Compute ratio: E[y]/(E[x] + ε)
            ratio = ey / (ex + self.epsilon)

            # EMA smoothing if enabled
            if self.ema_gamma < 1.0:
                ratio_s = self.ema_gamma * self.ratio_ema + (1 - self.ema_gamma) * ratio.mean()
                self.ratio_ema.copy_(ratio_s)
            else:
                ratio_s = ratio.mean()

            # Compute adaptive alpha
            self._alpha_cached = compute_alpha(self.alpha0, self.beta, ratio_s)

        # Update step count
        self.step_count.add_(1)

        # Apply residual connection with cached alpha
        alpha = self._alpha_cached if self._alpha_cached is not None else self.alpha0
        return x + alpha * y

    def extra_repr(self) -> str:
        """Return extra representation for debugging."""
        return (
            f"alpha0={self.alpha0.item():.4f}, beta={self.beta.item():.4f}, "
            f"ema_gamma={self.ema_gamma}, k_update={self.k_update}"
        )
