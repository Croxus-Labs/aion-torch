"""Benchmark crash test: AION vs Standard Transformer on extreme depth (GPU).

This script compares stability at extreme depth with identical hyperparameters
and reports mean±std over multiple seeds and crash-rate.

Usage:
    python examples/benchmark_crash_test_gpu.py

Output:
    - crash_test_data_gpu.csv: Training history (loss, gradients)

Notes:
    - Both models use the same optimizer/lr/clip.
    - We measure crash occurrence across seeds for fairness.
"""

import gc
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aion_torch import AionResidual

# ============================================================================
# Configuration
# ============================================================================
DEPTH = 600  # Deep enough to test stability while fitting in 8GB GPU memory
DIM = 128  # Small dimension to fit in VRAM
SEQ_LEN = 64
BATCH_SIZE = 2
NUM_STEPS = 150  # Increased to test long-term stability (3x longer)
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 10.0
NUM_SEEDS = 3  # Reduced for memory constraints (can increase if memory allows)

if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. Please use the CPU version or enable CUDA.")
DEVICE = torch.device("cuda")
COOLDOWN_SECONDS = 2
print(f"Using device: {DEVICE}")

# ============================================================================
# Standard Transformer
# ============================================================================


class StandardResidual(nn.Module):
    """Standard residual connection with Pre-LayerNorm (realistic transformer).

    Uses Pre-LayerNorm architecture for fair comparison with modern transformers.
    Even with LayerNorm, extreme depth (1000 layers) can cause gradient instability.
    """

    def __init__(self, dim: int):
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


class StandardTransformer(nn.Module):
    """1000-layer transformer with standard residual connections."""

    def __init__(self, depth: int, dim: int, seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(1000, dim)  # Dummy vocab size
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.layers = nn.ModuleList([StandardResidual(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.pos_embedding
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.output(x)
        return x


# ============================================================================
# AION Transformer
# ============================================================================


class AionResidualBlock(nn.Module):
    """Residual block using AION adaptive scaling with Pre-LayerNorm.

    Same structure as StandardResidual, but with AION instead of
    standard addition. Shows AION works with standard normalization
    and provides additional stability at extreme depth.
    """

    def __init__(self, dim: int):
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
        result: torch.Tensor = self.aion(residual, x)  # type: ignore[assignment]
        return result


class AionTransformer(nn.Module):
    """1000-layer transformer with AION adaptive residual connections."""

    def __init__(self, depth: int, dim: int, seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(1000, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.layers = nn.ModuleList([AionResidualBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.pos_embedding
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.output(x)
        return x


# ============================================================================
# Benchmark Execution
# ============================================================================


def _adjust_target_shape(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Adjust target shape to match output if needed."""
    if output.shape == target.shape:
        return target
    if len(output.shape) == 3 and len(target.shape) == 3:
        min_seq = min(output.shape[1], target.shape[1])
        return target[:, :min_seq, :]
    return target


def _compute_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm across all parameters."""
    grad_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
            param_count += 1
    return (grad_norm**0.5) if param_count > 0 else 0.0


def cooldown(seconds: int = COOLDOWN_SECONDS) -> None:
    """Brief pause between seeds to reset system state."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(seconds)


def _train_model_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dummy_input: torch.Tensor,
    dummy_target: torch.Tensor,
    step: int,
) -> tuple[float, float, bool]:
    """Train model for one step. Returns (loss, grad_norm, crashed)."""
    try:
        optimizer.zero_grad()
        output = model(dummy_input)
        dummy_target_adj = _adjust_target_shape(output, dummy_target)
        loss = criterion(output, dummy_target_adj)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] Step {step + 1}: FAILED - Loss is NaN/Inf")
            return float("nan"), float("nan"), True

        loss.backward()
        grad_norm = _compute_grad_norm(model)

        if math.isnan(grad_norm) or math.isinf(grad_norm):
            print(f"[ERROR] Step {step + 1}: FAILED - Gradients are NaN/Inf")
            return float("nan"), float("nan"), True

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        return loss.item(), grad_norm, False

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[ERROR] Step {step + 1}: FAILED - Out of Memory")
        else:
            print(f"[ERROR] Step {step + 1}: FAILED - RuntimeError: {e}")
        return float("nan"), float("nan"), True
    except (ValueError, TypeError, AttributeError) as e:
        print(f"[ERROR] Step {step + 1}: FAILED - Exception: {e}")
        return float("nan"), float("nan"), True


def _run_model_test(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dummy_input: torch.Tensor,
    dummy_target: torch.Tensor,
    model_name: str,
    expected_result: str,
) -> tuple[list[float], list[float], bool, int | None]:
    """Run training test for a model. Returns (losses, grads, crashed, crash_step)."""
    print("\n" + "=" * 80)
    print(f"Testing L={DEPTH} {model_name} (Expected: {expected_result})")
    print("=" * 80)

    losses = []
    grads = []
    crashed = False
    crash_step = None

    for step in range(NUM_STEPS):
        loss, grad_norm, step_crashed = _train_model_one_step(
            model, optimizer, criterion, dummy_input, dummy_target, step
        )

        if step_crashed:
            crashed = True
            crash_step = step + 1
            losses.extend([float("nan")] * (NUM_STEPS - step))
            grads.extend([float("nan")] * (NUM_STEPS - step))
            break

        losses.append(loss)
        grads.append(grad_norm)

        if (step + 1) % 25 == 0:
            print(
                f"  Step {step + 1:4d}/{NUM_STEPS}: Loss={loss:.6f}, GradNorm={grad_norm:.4f}"
            )

    if not crashed:
        print(f"[DONE] {model_name} completed {NUM_STEPS} steps successfully!")
        print(f"   Final loss: {losses[-1]:.6f}")
    else:
        print(f"[CRASH] {model_name} crashed at step {crash_step}")

    return losses, grads, crashed, crash_step


def run_crash_test():
    """Run the crash test comparing Standard vs AION."""
    print("=" * 80)
    print("AION-TORCH CRASH TEST")
    print("=" * 80)
    print(f"Depth: {DEPTH} layers | Dim: {DIM} | Seq: {SEQ_LEN} | Batch: {BATCH_SIZE}")
    print(f"Steps: {NUM_STEPS}")
    print("=" * 80)

    criterion = nn.MSELoss()
    dummy_input = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    dummy_target = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)

    all_losses_A, all_losses_B, all_grads_A, all_grads_B = [], [], [], []
    crashes_A, crashes_B = 0, 0
    for seed in range(NUM_SEEDS):
        if seed > 0:
            print(f"\n[Cooldown] Waiting {COOLDOWN_SECONDS}s before seed {seed + 1}...")
            cooldown()
        print(f"\n{'=' * 80}")
        print(f"SEED {seed + 1}/{NUM_SEEDS}")
        print(f"{'=' * 80}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model_A = StandardTransformer(depth=DEPTH, dim=DIM, seq_len=SEQ_LEN).to(DEVICE)
        model_B = AionTransformer(depth=DEPTH, dim=DIM, seq_len=SEQ_LEN).to(DEVICE)
        optimizer_A = torch.optim.AdamW(
            model_A.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )
        optimizer_B = torch.optim.AdamW(
            model_B.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        losses_A, grads_A, crashed_A, _ = _run_model_test(
            model_A,
            optimizer_A,
            criterion,
            dummy_input,
            dummy_target,
            "Standard Transformer",
            "—",
        )
        losses_B, grads_B, crashed_B, _ = _run_model_test(
            model_B,
            optimizer_B,
            criterion,
            dummy_input,
            dummy_target,
            "AION Transformer",
            "—",
        )
        crashes_A += int(crashed_A)
        crashes_B += int(crashed_B)
        all_losses_A.append(losses_A)
        all_grads_A.append(grads_A)
        all_losses_B.append(losses_B)
        all_grads_B.append(grads_B)

        del model_A, model_B, optimizer_A, optimizer_B
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def mean_over_seeds(series_list):
        out = []
        for t in range(NUM_STEPS):
            vals = [s[t] for s in series_list if t < len(s)]
            if not vals:
                out.append(float("nan"))
                continue
            m = sum(v for v in vals if not (isinstance(v, float) and math.isnan(v)))
            n = sum(1 for v in vals if not (isinstance(v, float) and math.isnan(v)))
            out.append(m / n if n > 0 else float("nan"))
        return out

    def std_over_seeds(series_list):
        out = []
        for t in range(NUM_STEPS):
            vals = [
                s[t]
                for s in series_list
                if t < len(s) and not (isinstance(s[t], float) and math.isnan(s[t]))
            ]
            if len(vals) < 2:
                out.append(0.0)
                continue
            mean_val = sum(vals) / len(vals)
            variance = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
            out.append(variance**0.5)
        return out

    losses_A_mean = mean_over_seeds(all_losses_A)
    losses_B_mean = mean_over_seeds(all_losses_B)
    grads_A_mean = mean_over_seeds(all_grads_A)
    grads_B_mean = mean_over_seeds(all_grads_B)
    losses_A_std = std_over_seeds(all_losses_A)
    losses_B_std = std_over_seeds(all_losses_B)
    grads_A_std = std_over_seeds(all_grads_A)
    grads_B_std = std_over_seeds(all_grads_B)
    crash_rate_A = crashes_A / NUM_SEEDS
    crash_rate_B = crashes_B / NUM_SEEDS

    print("\n" + "=" * 80)
    print("SUMMARY (aggregated over seeds)")
    print("=" * 80)
    print("Standard Transformer:")
    print(f"  Final Loss:     {losses_A_mean[-1]:.6f} ± {losses_A_std[-1]:.6f}")
    print(f"  Final GradNorm: {grads_A_mean[-1]:.4f} ± {grads_A_std[-1]:.4f}")
    print(f"  Crash-rate:     {crash_rate_A * 100:.1f}% over {NUM_SEEDS} seeds")
    print()
    print("AION Transformer:")
    print(f"  Final Loss:     {losses_B_mean[-1]:.6f} ± {losses_B_std[-1]:.6f}")
    print(f"  Final GradNorm: {grads_B_mean[-1]:.4f} ± {grads_B_std[-1]:.4f}")
    print(f"  Crash-rate:     {crash_rate_B * 100:.1f}% over {NUM_SEEDS} seeds")
    print("=" * 80)

    return (
        losses_A_mean,
        losses_B_mean,
        grads_A_mean,
        grads_B_mean,
        losses_A_std,
        losses_B_std,
        grads_A_std,
        grads_B_std,
    )


def _safe_value(val: float) -> float | str:
    """Convert NaN to string 'nan' for CSV."""
    if isinstance(val, float) and math.isnan(val):
        return "nan"
    return val


def save_csv_data(
    losses_a_mean,
    losses_b_mean,
    grads_a_mean,
    grads_b_mean,
    losses_a_std,
    losses_b_std,
    grads_a_std,
    grads_b_std,
):
    """Save results with mean and std to CSV file."""
    import csv

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "crash_test_data_gpu.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "loss_standard_mean",
                "loss_standard_std",
                "loss_aion_mean",
                "loss_aion_std",
                "grad_norm_standard_mean",
                "grad_norm_standard_std",
                "grad_norm_aion_mean",
                "grad_norm_aion_std",
            ]
        )
        for i in range(len(losses_a_mean)):
            writer.writerow(
                [
                    i + 1,
                    _safe_value(losses_a_mean[i]),
                    _safe_value(losses_a_std[i]),
                    _safe_value(losses_b_mean[i]),
                    _safe_value(losses_b_std[i]),
                    _safe_value(grads_a_mean[i]),
                    _safe_value(grads_a_std[i]),
                    _safe_value(grads_b_mean[i]),
                    _safe_value(grads_b_std[i]),
                ]
            )
    print(f"[DONE] Data saved to: {csv_path}")


def main():
    """Main execution."""
    try:
        (
            losses_A_mean,
            losses_B_mean,
            grads_A_mean,
            grads_B_mean,
            losses_A_std,
            losses_B_std,
            grads_A_std,
            grads_B_std,
        ) = run_crash_test()

        save_csv_data(
            losses_A_mean,
            losses_B_mean,
            grads_A_mean,
            grads_B_mean,
            losses_A_std,
            losses_B_std,
            grads_A_std,
            grads_B_std,
        )

        print("\n" + "=" * 80)
        print("[DONE] BENCHMARK COMPLETE")
        print("=" * 80)
        print("\nResults saved to:")
        print("  - examples/outputs/crash_test_data_gpu.csv")
        print("\nBenchmark complete.")

    except (RuntimeError, ValueError, TypeError, AttributeError) as e:
        print(f"\n[ERROR] Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
