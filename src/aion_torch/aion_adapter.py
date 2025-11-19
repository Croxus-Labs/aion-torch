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

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Apply adaptive residual connection.

        Args:
            x: Input tensor (residual branch input)
            y: Transform output tensor (e.g., from FFN, attention)
            return_stats: Whether to return internal statistics (alpha, ratio)

        Returns:
            Combined output: x + α · y where α is adaptive
            If return_stats is True, returns (output, stats_dict)

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
        current_ratio_val = 0.0
        if self.training:
            # Compute energies
            ex = energy(x, dim=-1, keepdim=False)  # [B, T] or [B] depending on dims
            ey = energy(y, dim=-1, keepdim=False)

            # Compute ratio: E[y]/(E[x] + ε)
            ratio = ey / (ex + self.epsilon)

            # Calculate mean ratio for this batch (scalar)
            ratio_mean = ratio if ratio.ndim == 0 else ratio.mean()

            # Distributed synchronization: Average ratio across all ranks
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(ratio_mean, op=torch.distributed.ReduceOp.AVG)

            current_ratio_val = ratio_mean.item()

            # EMA smoothing if enabled
            if self.ema_gamma < 1.0:
                # Detach ratio_ema to prevent gradient history accumulation
                ratio_s = self.ema_gamma * self.ratio_ema.detach() + (1 - self.ema_gamma) * ratio_mean
                # Update running average with detached value
                self.ratio_ema = ratio_s.detach()
            else:
                ratio_s = ratio_mean

            # Compute adaptive alpha
            # Use assignment instead of copy_() to avoid in-place graph extension
            # which causes "backward through graph a second time" errors
            self.alpha_cached = compute_alpha(self.alpha0, self.beta, ratio_s)

            # Update step count for tracking
            self.step_count.add_(1)

        # Apply residual connection with cached alpha
        out = x + self.alpha_cached * y

        if return_stats:
            stats = {
                "alpha": self.alpha_cached.item(),
                "ratio": current_ratio_val if self.training else self.ratio_ema.item(),
            }
            return out, stats

        return out

    def extra_repr(self) -> str:
        """Return extra representation for debugging."""
        return f"alpha0={self.alpha0.item():.4f}, beta={self.beta.item():.4f}, " f"ema_gamma={self.ema_gamma}"
