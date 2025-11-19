"""Overhead benchmark: Standard residual vs AION residual (CPU version).

This script measures the training-step runtime overhead of replacing a
standard Pre-LayerNorm residual block with the AION adaptive residual block.

Usage:
    python examples/benchmark_overhead_test_cpu.py

Output:
    Prints average step time, standard deviation, and steps/second for both
    implementations alongside relative overhead statistics.
"""

from __future__ import annotations

import csv
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from aion_torch import AionResidual

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 8
SEQ_LEN = 128
DIM = 512
WARMUP_STEPS = 20
MEASURE_STEPS = 150
DEVICE = torch.device("cpu")
COOLDOWN_SECONDS = 5


@dataclass
class BenchmarkResult:
    name: str
    mean_step_time: float
    std_step_time: float
    steps_per_second: float


def maybe_sync_device() -> None:
    """Synchronize CUDA device if available."""

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def cooldown(seconds: int = COOLDOWN_SECONDS) -> None:
    """Brief pause to cool down between runs and reset caches."""
    maybe_sync_device()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(seconds)


class StandardResidualBlock(nn.Module):
    """Single FFN block with Pre-LayerNorm residual connection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        return residual + x


class AionResidualBlock(nn.Module):
    """Single FFN block with AION adaptive residual connection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.aion = AionResidual(alpha0=0.1, beta=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        return self.aion(residual, x)  # type: ignore


def build_model(block_cls: type[nn.Module]) -> nn.Module:
    """Constructs a tiny 4-layer transformer-like stack using the block."""

    layers = []
    for _ in range(4):
        layers.append(block_cls(DIM))
    model = nn.Sequential(*layers)
    return model.to(DEVICE)


def benchmark_model(model: nn.Module, name: str) -> tuple[BenchmarkResult, list[float]]:
    """Benchmark average step time for a simple training loop.

    Returns:
        BenchmarkResult with statistics and list of all measured times.
    """

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    criterion = nn.MSELoss()

    x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)
    target = torch.randn_like(x)

    times: list[float] = []

    for step in range(WARMUP_STEPS + MEASURE_STEPS):
        optimizer.zero_grad(set_to_none=True)

        maybe_sync_device()
        start = time.perf_counter()

        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        maybe_sync_device()
        end = time.perf_counter()

        if step >= WARMUP_STEPS:
            times.append(end - start)

    mean_time = statistics.mean(times)
    std_time = statistics.pstdev(times)
    steps_per_sec = 1.0 / mean_time

    result = BenchmarkResult(
        name=name,
        mean_step_time=mean_time,
        std_step_time=std_time,
        steps_per_second=steps_per_sec,
    )
    return result, times


def print_summary(standard: BenchmarkResult, aion: BenchmarkResult) -> None:
    """Print benchmark results and relative overhead."""

    overhead_pct = (aion.mean_step_time / standard.mean_step_time - 1.0) * 100.0

    print("=" * 80)
    print("AION Overhead Benchmark (CPU version)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE} | Sequence Length: {SEQ_LEN} | Dim: {DIM}")
    print(f"Measured Steps: {MEASURE_STEPS} (warmup {WARMUP_STEPS} steps)")
    print("=" * 80)

    for result in (standard, aion):
        print(f"{result.name}:")
        print(f"  Mean step time:   {result.mean_step_time * 1000:.3f} ms")
        print(f"  Step time stddev: {result.std_step_time * 1000:.3f} ms")
        print(f"  Steps / second:   {result.steps_per_second:.2f}")
        print("-" * 80)

    print("Relative Overhead (AION vs Standard):")
    print(f"  {overhead_pct:+.2f}% per training step")
    print("=" * 80)


def save_csv_data(standard: BenchmarkResult, aion: BenchmarkResult) -> None:
    """Save benchmark results to CSV file."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "overhead_test_data_cpu.csv"

    overhead_pct = (aion.mean_step_time / standard.mean_step_time - 1.0) * 100.0

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "mean_step_time_ms",
                "std_step_time_ms",
                "steps_per_second",
                "overhead_percent",
            ]
        )
        writer.writerow(
            [
                standard.name,
                f"{standard.mean_step_time * 1000:.6f}",
                f"{standard.std_step_time * 1000:.6f}",
                f"{standard.steps_per_second:.6f}",
                "0.00",
            ]
        )
        writer.writerow(
            [
                aion.name,
                f"{aion.mean_step_time * 1000:.6f}",
                f"{aion.std_step_time * 1000:.6f}",
                f"{aion.steps_per_second:.6f}",
                f"{overhead_pct:.2f}",
            ]
        )

    print(f"\nSaved data to: {csv_path}")


def main() -> int:
    torch.manual_seed(42)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)

    print("Preparing models...")
    standard_model = build_model(StandardResidualBlock)
    aion_model = build_model(AionResidualBlock)

    print("Benchmarking standard residual (run 1)...")
    _, standard_times_1 = benchmark_model(standard_model, "Standard Residual")
    cooldown()
    print("Benchmarking AION residual (run 1)...")
    _, aion_times_1 = benchmark_model(aion_model, "AION Residual")

    print("Preparing fresh models for second run...")
    standard_model2 = build_model(StandardResidualBlock)
    aion_model2 = build_model(AionResidualBlock)

    cooldown()
    print("Benchmarking AION residual (run 2)...")
    _, aion_times_2 = benchmark_model(aion_model2, "AION Residual")
    cooldown()
    print("Benchmarking standard residual (run 2)...")
    _, standard_times_2 = benchmark_model(standard_model2, "Standard Residual")

    all_standard_times = standard_times_1 + standard_times_2
    all_aion_times = aion_times_1 + aion_times_2

    standard_result = BenchmarkResult(
        name="Standard Residual",
        mean_step_time=statistics.mean(all_standard_times),
        std_step_time=statistics.pstdev(all_standard_times),
        steps_per_second=1.0 / statistics.mean(all_standard_times),
    )

    aion_result = BenchmarkResult(
        name="AION Residual",
        mean_step_time=statistics.mean(all_aion_times),
        std_step_time=statistics.pstdev(all_aion_times),
        steps_per_second=1.0 / statistics.mean(all_aion_times),
    )

    print_summary(standard_result, aion_result)
    save_csv_data(standard_result, aion_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
