# AION-Torch: Adaptive Input/Output Normalization

[![PyPI version](https://img.shields.io/pypi/v/aion-torch.svg)](https://pypi.org/project/aion-torch/)
[![PyPI downloads](https://img.shields.io/pypi/dm/aion-torch.svg)](https://pypi.org/project/aion-torch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

**AION-Torch** is a PyTorch library that implements **Adaptive Input/Output Normalization (AION)**, a method for stabilizing deep neural networks. AION automatically adjusts residual connections to prevent vanishing and exploding gradients, enabling stable training of very deep networks with minimal configuration.

---

## 🚀 Features

- **Adaptive Residual Scaling**: Automatically adjusts residual connection strength based on signal statistics
- **Stable Deep Training**: Prevents vanishing/exploding gradients even in networks with 1000+ layers
- **Drop-in Replacement**: Works with any architecture using residual connections (Transformers, ResNets, etc.)
- **Distributed Ready**: Fully supports DDP with synchronized statistics across all GPUs
- **Zero Config**: Sensible defaults work out-of-the-box, no hyperparameter tuning needed

## 📦 Installation

### From PyPI
```bash
pip install aion-torch
```

## ⚡ Quick Start

### 1. The `AionBlock` (Recommended)
The easiest way to use AION is to replace your standard residual blocks with `AionBlock`. It implements the **Pre-LayerNorm** pattern augmented with AION scaling.

```python
import torch
import torch.nn as nn
from aion_torch import AionBlock

# Define your transformation layer (e.g., Attention or MLP)
mlp_layer = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512)
)

# Wrap it in an AionBlock
# Structure: x + alpha * layer(norm(x))
block = AionBlock(layer=mlp_layer, dim=512)

# Forward pass
x = torch.randn(8, 128, 512)
output = block(x)
```

### 2. Low-Level `AionResidual`
For custom architectures, you can use the `AionResidual` adapter directly.

```python
from aion_torch import AionResidual

class MyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Linear(dim, dim)
        # Initialize AION adapter
        self.aion = AionResidual(alpha0=0.1, beta=0.05)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        y = self.ffn(x_norm)

        # Apply adaptive residual connection
        # Formula: x + alpha * y
        return self.aion(residual, y)
```

## 🧠 How It Works

AION adaptively scales residual connections using a simple but effective formula:

$$
\alpha = \frac{\alpha_0}{1 + \beta \cdot \text{ratio}}
$$

where `ratio` measures the relative magnitude of the transformation output compared to the input. When the network becomes unstable (high ratio), AION automatically reduces the scaling factor. When stable (low ratio), it uses a stronger connection.

**Key insight**: By maintaining balanced signal propagation, AION ensures gradients flow stably through arbitrarily deep networks without exponential growth or decay.

*For the theoretical foundation and mathematical proofs, DM me.*

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) (coming soon) and check out the issues.

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add some amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ x Abbasagha Babayev</sub>
</div>
