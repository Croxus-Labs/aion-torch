"""AION: Adaptive Input/Output Normalization for deep neural networks.

AION provides adaptive scaling of residual connections based on energy ratios,
improving training stability in very deep networks.

Basic usage:
    >>> import torch
    >>> from aion_torch import AionResidual
    >>>
    >>> layer = AionResidual(alpha0=0.1, beta=0.05)
    >>> x = torch.randn(8, 512)
    >>> y = torch.randn(8, 512)
    >>> out = layer(x, y)

Advanced usage with registry:
    >>> from aion_torch import register_adapter, make_adapter
    >>> register_adapter("aion", AionResidual)
    >>> layer = make_adapter("aion", alpha0=0.1, beta=0.05)
"""

__version__ = "0.2.0"

from .aion_adapter import AionResidual
from .alpha import compute_alpha
from .energy import energy
from .registry import list_adapters, make_adapter, register_adapter

# Auto-register adapters
register_adapter("aion", AionResidual)

__all__ = [
    "AionResidual",
    "energy",
    "compute_alpha",
    "register_adapter",
    "make_adapter",
    "list_adapters",
    "__version__",
]
