"""Visualize benchmark results from CSV data.

This script reads the benchmark CSV files and generates PNG charts
showing the comparison between Standard and AION transformers.

Usage:
    python examples/visualize_results.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_float(value):
    """Convert value to float, handling 'nan' strings."""
    if isinstance(value, str) and value.lower() == "nan":
        return float("nan")
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def plot_crash_test_results():
    """Plot crash test results: loss and gradient norms over training steps."""
    csv_path = OUTPUT_DIR / "crash_test_data_gpu.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping crash test visualization.")
        return

    df = pd.read_csv(csv_path)

    for col in df.columns:
        if col != "step":
            df[col] = df[col].apply(safe_float)
    
    steps = df["step"].values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Loss comparison
    ax1.plot(
        steps,
        df["loss_standard_mean"],
        label="Standard Transformer",
        color="#d62728",
        linewidth=2,
    )
    ax1.fill_between(
        steps,
        df["loss_standard_mean"] - df["loss_standard_std"],
        df["loss_standard_mean"] + df["loss_standard_std"],
        alpha=0.2,
        color="#d62728",
    )

    aion_valid = ~np.isnan(df["loss_aion_mean"])
    if aion_valid.any():
        last_valid_idx = np.where(aion_valid)[0][-1]
        aion_steps = steps[: last_valid_idx + 1]
        aion_loss_mean = df["loss_aion_mean"].values[: last_valid_idx + 1]
        aion_loss_std = df["loss_aion_std"].values[: last_valid_idx + 1]

        ax1.plot(
            aion_steps,
            aion_loss_mean,
            label="AION Transformer",
            color="#2ca02c",
            linewidth=2,
        )
        ax1.fill_between(
            aion_steps,
            aion_loss_mean - aion_loss_std,
            aion_loss_mean + aion_loss_std,
            alpha=0.2,
            color="#2ca02c",
        )

        if last_valid_idx < len(steps) - 1:
            crash_step = steps[last_valid_idx + 1]
            ax1.axvline(
                x=crash_step,
                color="#2ca02c",
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
                label=f"AION OOM at step {crash_step}",
            )

    ax1.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss (MSE)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Crash Test: Loss Comparison (600-layer Transformer)", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2.plot(
        steps,
        df["grad_norm_standard_mean"],
        label="Standard Transformer",
        color="#d62728",
        linewidth=2,
    )
    ax2.fill_between(
        steps,
        df["grad_norm_standard_mean"] - df["grad_norm_standard_std"],
        df["grad_norm_standard_mean"] + df["grad_norm_standard_std"],
        alpha=0.2,
        color="#d62728",
    )
    
    if aion_valid.any():
        aion_grad_mean = df["grad_norm_aion_mean"].values[: last_valid_idx + 1]
        aion_grad_std = df["grad_norm_aion_std"].values[: last_valid_idx + 1]
        
        ax2.plot(
            aion_steps,
            aion_grad_mean,
            label="AION Transformer",
            color="#2ca02c",
            linewidth=2,
        )
        ax2.fill_between(
            aion_steps,
            aion_grad_mean - aion_grad_std,
            aion_grad_mean + aion_grad_std,
            alpha=0.2,
            color="#2ca02c",
        )
        
        if last_valid_idx < len(steps) - 1:
            ax2.axvline(
                x=crash_step,
                color="#2ca02c",
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
            )
    
    ax2.set_xlabel("Training Step", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Gradient Norm", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Crash Test: Gradient Norm Comparison", fontsize=14, fontweight="bold"
    )
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "crash_test_results_gpu.png"
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved crash test chart: {output_path}")
    plt.close()


def plot_overhead_results():
    """Plot overhead benchmark results."""
    csv_path = OUTPUT_DIR / "overhead_test_data_gpu.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping overhead visualization.")
        return

    df = pd.read_csv(csv_path)

    methods = df["method"].values
    mean_times = df["mean_step_time_ms"].values
    std_times = df["std_step_time_ms"].values
    overhead_pct = float(df[df["method"] == "AION Residual"]["overhead_percent"].values[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = ["#d62728", "#2ca02c"]
    bars = ax.bar(
        methods,
        mean_times,
        yerr=std_times,
        capsize=10,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    for i, (bar, mean, std) in enumerate(zip(bars, mean_times, std_times)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.5,
            f"{mean:.2f} ms\n(±{std:.2f})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    
    ax.set_ylabel("Mean Step Time (ms)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Overhead Benchmark: AION vs Standard Residual\n(Unoptimized AION baseline, overhead: +{overhead_pct:.2f}%)",
        fontsize=14,
        fontweight="bold",
        pad=24,
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "overhead_test_results_gpu.png"
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved overhead chart: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("Generating benchmark visualizations...")
    print("=" * 60)
    
    plot_crash_test_results()
    plot_overhead_results()
    
    print("=" * 60)
    print("[OK] All visualizations generated successfully!")
    print(f"\nCharts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
