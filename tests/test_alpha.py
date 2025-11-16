"""Tests for alpha computation."""

import torch

from aion_torch.alpha import compute_alpha  # type: ignore


class TestComputeAlpha:
    """Test suite for compute_alpha function."""

    def test_basic_computation(self) -> None:
        """Test basic alpha computation with simple values."""
        alpha0 = torch.tensor(0.1)
        beta = torch.tensor(0.05)
        ratio_s = torch.tensor(0.5)

        alpha = compute_alpha(alpha0, beta, ratio_s)

        # α = α₀ / (1 + β · ratio_s) = 0.1 / (1 + 0.05 * 0.5) = 0.1 / 1.025
        expected = 0.1 / 1.025
        assert torch.allclose(alpha, torch.tensor(expected), atol=1e-6)

    def test_zero_ratio(self) -> None:
        """Test alpha computation when ratio is zero."""
        alpha0 = torch.tensor(0.1)
        beta = torch.tensor(0.05)
        ratio_s = torch.tensor(0.0)

        alpha = compute_alpha(alpha0, beta, ratio_s)

        # α = α₀ / (1 + 0) = α₀
        assert torch.allclose(alpha, alpha0)

    def test_large_ratio(self) -> None:
        """Test alpha computation with large ratio (strong dampening)."""
        alpha0 = torch.tensor(0.1)
        beta = torch.tensor(0.05)
        ratio_s = torch.tensor(100.0)

        alpha = compute_alpha(alpha0, beta, ratio_s)

        # α should be much smaller than α₀
        assert alpha < alpha0
        expected = 0.1 / (1 + 0.05 * 100)  # = 0.1 / 6.0
        assert torch.allclose(alpha, torch.tensor(expected), atol=1e-6)

    def test_zero_beta(self) -> None:
        """Test alpha computation when beta is zero (no adaptation)."""
        alpha0 = torch.tensor(0.1)
        beta = torch.tensor(0.0)
        ratio_s = torch.tensor(100.0)

        alpha = compute_alpha(alpha0, beta, ratio_s)

        # α = α₀ / (1 + 0) = α₀
        assert torch.allclose(alpha, alpha0)

    def test_different_dtypes(self) -> None:
        """Test that function works with different dtypes."""
        alpha0 = torch.tensor(0.1, dtype=torch.float64)
        beta = torch.tensor(0.05, dtype=torch.float64)
        ratio_s = torch.tensor(0.5, dtype=torch.float64)

        alpha = compute_alpha(alpha0, beta, ratio_s)

        assert alpha.dtype == torch.float64
        expected = 0.1 / 1.025
        assert torch.allclose(alpha, torch.tensor(expected, dtype=torch.float64))

    def test_batch_computation(self) -> None:
        """Test alpha computation with batched ratio_s."""
        alpha0 = torch.tensor(0.1)
        beta = torch.tensor(0.05)
        ratio_s = torch.tensor([0.5, 1.0, 2.0])

        alpha = compute_alpha(alpha0, beta, ratio_s)

        assert alpha.shape == ratio_s.shape
        expected = torch.tensor([0.1 / 1.025, 0.1 / 1.05, 0.1 / 1.1])
        assert torch.allclose(alpha, expected, atol=1e-6)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through alpha computation."""
        alpha0 = torch.tensor(0.1, requires_grad=True)
        beta = torch.tensor(0.05, requires_grad=True)
        ratio_s = torch.tensor(0.5)

        alpha = compute_alpha(alpha0, beta, ratio_s)
        loss = alpha.sum()
        loss.backward()

        assert alpha0.grad is not None
        assert beta.grad is not None
        assert alpha0.grad.abs() > 0
        assert beta.grad.abs() > 0
