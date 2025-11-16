"""Energy computation with fp32 accumulation.

This module provides energy calculation E[t] = mean(t²) with numerical
stability guarantees through fp32 accumulation.
"""

import torch


def energy(t: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    """Compute mean squared energy E[t] = mean(t²) with fp32 accumulation.

    Args:
        t: Input tensor of any shape
        dim: Dimension along which to compute mean (default: -1, last dimension)
        keepdim: Whether to keep the reduced dimension (default: True)

    Returns:
        Energy tensor E[t] = mean(t²) in original dtype

    Notes:
        - Forces fp32 accumulation for numerical stability
        - Returns result in original dtype of input tensor

    Example:
        >>> x = torch.randn(8, 128, 512)  # [B, T, D]
        >>> ex = energy(x, dim=-1, keepdim=True)  # [B, T, 1]
    """
    # Convert to fp32 for accumulation to ensure numerical stability
    acc = t.float()
    # Compute mean of squares
    e = (acc * acc).mean(dim=dim, keepdim=keepdim)
    # Return in original dtype
    return e.to(dtype=t.dtype)
