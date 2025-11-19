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
        epsilon: float = 1e-8,
    ):
        """Initialize AION residual adapter.

        Args:
            alpha0: Base scaling parameter (learnable). Must be positive.
            beta: Adaptation coefficient (learnable). Must be non-negative.
            ema_gamma: EMA smoothing factor for ratio_s (1.0 = no smoothing).
                Must be in [0, 1].
            epsilon: Small constant for numerical stability. Must be positive.

        Raises:
            ValueError: If any parameter is out of valid range.

        Note:
            Alpha updates every forward pass in training mode to ensure correct
            behavior in distributed training (DataParallel/DDP).
        """
        super().__init__()

        # Input validation
        if alpha0 <= 0:
            raise ValueError(f"alpha0 must be positive, got {alpha0}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if not (0 <= ema_gamma <= 1):
            raise ValueError(f"ema_gamma must be in [0, 1], got {ema_gamma}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Learnable parameters
        self.alpha0 = nn.Parameter(torch.tensor(alpha0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

        # Configuration
        self.ema_gamma = ema_gamma
        self.epsilon = epsilon

        # State - register alpha_cached as buffer for proper state dict handling
        self.register_buffer("ratio_ema", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.int32))
        # Initialize alpha_cached as buffer (will be updated during forward)
        self.register_buffer("alpha_cached", torch.tensor(alpha0, dtype=torch.float32))

        # Type hints for buffers (for static type checkers)
        self.ratio_ema: torch.Tensor
        self.step_count: torch.Tensor
        self.alpha_cached: torch.Tensor

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply adaptive residual connection.

        Args:
            x: Input tensor (residual branch input)
            y: Transform output tensor (e.g., from FFN, attention)

        Returns:
            Combined output: x + α · y where α is adaptive

        Raises:
            ValueError: If input tensors have mismatched shapes or are empty.
        """
        # Input validation - check empty tensors first
        if x.numel() == 0:
            raise ValueError("Input tensor x cannot be empty")
        if y.numel() == 0:
            raise ValueError("Input tensor y cannot be empty")
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match: x.shape={x.shape}, y.shape={y.shape}")

        # Always update alpha in training mode
        # This ensures correct behavior in distributed training (DataParallel/DDP)
        if self.training:
            # Compute energies
            ex = energy(x, dim=-1, keepdim=False)  # [B, T] or [B] depending on dims
            ey = energy(y, dim=-1, keepdim=False)

            # Compute ratio: E[y]/(E[x] + ε)
            ratio = ey / (ex + self.epsilon)

            # EMA smoothing if enabled
            if self.ema_gamma < 1.0:
                # Handle scalar case for ratio.mean()
                ratio_mean = ratio if ratio.ndim == 0 else ratio.mean()
                ratio_s = self.ema_gamma * self.ratio_ema + (1 - self.ema_gamma) * ratio_mean
                self.ratio_ema.copy_(ratio_s)
            else:
                # Handle scalar case
                ratio_s = ratio if ratio.ndim == 0 else ratio.mean()

            # Compute adaptive alpha
            self.alpha_cached.copy_(compute_alpha(self.alpha0, self.beta, ratio_s))

            # Update step count for tracking
            self.step_count.add_(1)

        # Apply residual connection with cached alpha
        return x + self.alpha_cached * y

    def extra_repr(self) -> str:
        """Return extra representation for debugging."""
        return f"alpha0={self.alpha0.item():.4f}, beta={self.beta.item():.4f}, " f"ema_gamma={self.ema_gamma}"
