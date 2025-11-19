"""Simple demonstration of AION adaptive residual connection.

This script shows how to use AION in practice and visualizes
the adaptive alpha parameter over training steps.
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


try:
    from aion_torch import AionResidual
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(ROOT))
    from aion_torch import AionResidual


def _get_alpha_cached(adapter: AionResidual) -> float:
    """Return cached alpha value (helper for demo logging)."""
    return float(adapter.alpha_cached.item())


def simple_demo():
    """Basic AION usage demonstration."""
    print("=" * 60)
    print("AION Simple Demo")
    print("=" * 60)

    layer = AionResidual(
        alpha0=0.1,
        beta=0.05,
        ema_gamma=0.99,
    )

    print(f"\nLayer config: {layer}")
    print(f"Initial alpha0: {layer.alpha0.item():.4f}")
    print(f"Initial beta: {layer.beta.item():.4f}")

    print("\nRunning forward passes...")

    batch_size, seq_len, dim = 8, 128, 512

    for step in range(20):
        x = torch.randn(batch_size, seq_len, dim)
        y = torch.randn(batch_size, seq_len, dim)

        out = layer(x, y)

        if step % 5 == 0:
            alpha = _get_alpha_cached(layer)
            ratio_ema = layer.ratio_ema.item()
            out_energy = out.pow(2).mean().item()
            print(
                f"Step {step:2d}: alpha={alpha:.6f}, ratio_ema={ratio_ema:.6f}, "
                f"out_energy={out_energy:.6f}"
            )

    print("\n[DONE] Demo completed successfully!")


def training_demo():
    """Demonstration of AION in a simple training loop."""
    print("\n" + "=" * 60)
    print("AION Training Demo")
    print("=" * 60)

    class SimpleModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )
            self.aion = AionResidual(alpha0=0.1, beta=0.05)
            self.ln = nn.LayerNorm(dim)

        def forward(self, x):
            x_norm = self.ln(x)
            y = self.ffn(x_norm)
            x = self.aion(x, y)
            return x

    dim = 256
    model = SimpleModel(dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nModel: {dim}-dim with AION residual")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining for 50 steps...")

    losses = []
    alphas = []

    for step in range(50):
        x = torch.randn(8, 64, dim)
        target = torch.randn(8, 64, dim)

        out = model(x)
        loss = nn.functional.mse_loss(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        alpha = _get_alpha_cached(model.aion)
        alphas.append(alpha)

        if step % 10 == 0:
            print(f"Step {step:2d}: loss={loss.item():.6f}, alpha={alpha:.6f}")

    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Final alpha: {alphas[-1]:.6f}")
    print(f"Alpha range: [{min(alphas):.6f}, {max(alphas):.6f}]")

    print("\n[DONE] Training demo completed successfully!")


def export_metrics_demo():
    """Demonstration with CSV export of metrics."""
    print("\n" + "=" * 60)
    print("AION Metrics Export Demo")
    print("=" * 60)

    layer = AionResidual(alpha0=0.1, beta=0.05, ema_gamma=0.99)

    metrics = []

    print("\nRunning 100 steps and collecting metrics...")

    for step in range(100):
        x = torch.randn(4, 64)
        y = torch.randn(4, 64)

        out = layer(x, y)

        alpha = _get_alpha_cached(layer)
        ratio_ema = layer.ratio_ema.item()
        out_energy = out.pow(2).mean().item()

        metrics.append(
            {
                "step": step,
                "alpha": alpha,
                "ratio_ema": ratio_ema,
                "alpha0": layer.alpha0.item(),
                "beta": layer.beta.item(),
                "out_energy": out_energy,
            }
        )

    output_file = Path(__file__).parent / "aion_metrics.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "alpha",
                "ratio_ema",
                "alpha0",
                "beta",
                "out_energy",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\n[DONE] Metrics exported to: {output_file}")
    print(f"  Total steps: {len(metrics)}")
    print(
        f"  Alpha range: [{min(m['alpha'] for m in metrics):.6f}, {max(m['alpha'] for m in metrics):.6f}]"
    )


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AION Demo Suite")
    print("=" * 60)
    simple_demo()
    training_demo()
    export_metrics_demo()

    print("\n" + "=" * 60)
    print("[DONE] All demos completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
