"""Alpha computation for AION adaptive scaling.

This module provides the core AION formula:
    α = α₀ / (1 + β · ratio_s)
where ratio_s = E[y]/(E[x] + ε) is optionally EMA-smoothed.
"""

import torch


def compute_alpha(alpha0: torch.Tensor, beta: torch.Tensor, ratio_s: torch.Tensor) -> torch.Tensor:
    """Compute AION adaptive scaling parameter.

    Formula: α = α₀ / (1 + β · ratio_s)

    Args:
        alpha0: Base scaling parameter (learnable)
        beta: Adaptation coefficient (learnable)
        ratio_s: Energy ratio E[y]/(E[x] + ε), optionally EMA-smoothed

    Returns:
        Adaptive alpha scaling factor

    Notes:
        - All inputs should be scalars or broadcastable
        - ratio_s is typically computed as: Ey / (Ex + epsilon)
        - Can be smoothed via EMA: ratio_ema = γ*ratio_ema + (1-γ)*ratio

    Example:
        >>> alpha0 = torch.tensor(0.1)
        >>> beta = torch.tensor(0.05)
        >>> ratio_s = torch.tensor(0.5)
        >>> alpha = compute_alpha(alpha0, beta, ratio_s)
    """
    return alpha0 / (1.0 + beta * ratio_s)
