# AION-Torch

> **[WARNING] Alpha Version:** This library is currently in alpha. APIs may change without notice. Use at your own risk.

**Adaptive Input/Output Normalization for deep neural networks.** AION dynamically adjusts residual connection scaling for stable training of extremely deep networks.

## What is AION?

AION (Adaptive Input/Output Normalization) is an adaptive residual scaling layer
that keeps the energy of residual branches in balance. Instead of using a fixed
scale for `x + y`, AION dynamically adjusts `α` in `x + α·y` based on the input
and output energies. This stabilizes very deep networks (hundreds of layers)
and improves convergence without manual tuning.

## The Proof

### Crash Test Results (600-layer Transformer, GPU)

**AION demonstrates superior numerical stability and faster convergence.**

![AION vs Standard Transformer Crash Test](https://raw.githubusercontent.com/Croxus-Labs/aion-torch/main/examples/outputs/crash_test_results_gpu.png)

_600-layer transformer test on GPU: Both models completed all 150 training steps successfully. AION Transformer achieved significantly lower loss (0.0011 ± 0.0003) and more stable gradients compared to Standard Transformer (0.0075 ± 0.0015)._

**Benchmark Methodology:**

Both models use **Pre-LayerNorm architecture** (normalization before the feedforward network), which is the standard practice in modern transformers (GPT, BERT, etc.). Pre-LayerNorm enables standard transformers to work at deep depths by normalizing activations before transformation, helping maintain stable gradient flow. We tested **600 layers** to demonstrate AION's advantages at extreme depth while ensuring both models complete the full training run without memory constraints. This makes the comparison fair—both models use the same modern best practices, and AION still demonstrates superior stability and convergence speed even at these extreme depths.

**Key Findings:**

- **Standard Transformer**: Completed all 150 steps, final loss: 0.0075 ± 0.0015, crash rate: 0%
- **AION Transformer**: Completed all 150 steps, final loss: 0.0011 ± 0.0003, crash rate: 0%
- **Gradient Stability**: AION maintained more stable and lower gradient norms (0.0135 ± 0.0033) vs Standard (0.0665 ± 0.0116)
- **Training Efficiency**: AION achieved ~7x lower final loss, demonstrating significantly faster convergence

These results suggest that AION can improve numerical stability and convergence
speed at extreme depths (600 layers), even on top of modern Pre-LayerNorm
architectures.

## Installation

Install from PyPI:

```bash
pip install aion-torch
```

Or install in development mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from aion_torch import AionResidual

# Create AION layer
layer = AionResidual(alpha0=0.1, beta=0.05)

# Use in residual connection
x = torch.randn(8, 128, 512)  # [batch, seq, features]
y = torch.randn(8, 128, 512)  # Output from FFN/attention
out = layer(x, y)             # Adaptive residual: x + α·y
```

### Overhead Benchmark Results (GPU)

**AION adds ~36% computational overhead per training step.**

![Overhead Benchmark Results](https://raw.githubusercontent.com/Croxus-Labs/aion-torch/main/examples/outputs/overhead_test_results_gpu.png)

_Benchmark configuration: 4-layer transformer, batch size 8, sequence length 128, dimension 512. Results averaged over 150 training steps (after 20 warmup steps)._

**Performance Metrics (Unoptimized Baseline):**

- **Standard Residual**: 9.79 ms/step (102.11 steps/sec)
- **AION Residual**: 13.36 ms/step (74.84 steps/sec)
- **Overhead**: +36.44% per training step

The overhead comes from AION's adaptive scaling calculations, which provide the
stability benefits shown in the crash test.

There are several ways to reduce this cost in practice:

- **Gradient accumulation**: accumulate gradients over multiple batches to
  amortize the per-batch overhead.
- **Engineering optimizations**: fusing operations, reusing statistics, or using
  lower precision for energy tracking. With careful optimization, we expect the
  overhead to be reduced to below ~5% in production setups.

Note: Alpha updates every forward pass in training mode to ensure correct
behavior in distributed training (DataParallel/DDP).

## Features

- **Adaptive scaling**: Automatically adjusts to network dynamics
- **Training stability**: Prevents gradient explosion and vanishing
- **Deep network support**: Works with networks of any depth
- **Faster convergence**: Achieves lower loss faster than standard residuals
- **PyTorch 2.0+**: Fully compatible with modern PyTorch

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Format code
make format

# Run linting
make lint

# Run tests
make test

# Install pre-commit hooks
make pre-commit-install
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

**Note:** This is an Alpha version. APIs may change without notice. Use at your own risk.

## Author

Abbasagha Babayev
