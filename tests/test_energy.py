"""Tests for energy computation."""

import torch

from aion_torch.energy import energy  # type: ignore


class TestEnergy:
    """Test suite for energy function."""

    def test_basic_energy(self) -> None:
        """Test basic energy computation."""
        x = torch.tensor([1.0, 2.0, 3.0])
        e = energy(x, dim=-1, keepdim=False)

        # E[x] = mean(x²) = (1 + 4 + 9) / 3 = 14/3
        expected = torch.tensor(14.0 / 3.0)
        assert torch.allclose(e, expected, atol=1e-6)

    def test_keepdim_true(self) -> None:
        """Test energy computation with keepdim=True."""
        x = torch.randn(4, 8, 16)
        e = energy(x, dim=-1, keepdim=True)

        assert e.shape == (4, 8, 1)
        assert e.dtype == x.dtype

    def test_keepdim_false(self) -> None:
        """Test energy computation with keepdim=False."""
        x = torch.randn(4, 8, 16)
        e = energy(x, dim=-1, keepdim=False)

        assert e.shape == (4, 8)
        assert e.dtype == x.dtype

    def test_different_dims(self) -> None:
        """Test energy computation along different dimensions."""
        x = torch.randn(4, 8, 16)

        e0 = energy(x, dim=0, keepdim=False)
        e1 = energy(x, dim=1, keepdim=False)
        e2 = energy(x, dim=2, keepdim=False)

        assert e0.shape == (8, 16)
        assert e1.shape == (4, 16)
        assert e2.shape == (4, 8)

    def test_zero_tensor(self) -> None:
        """Test energy of zero tensor."""
        x = torch.zeros(4, 8, 16)
        e = energy(x, dim=-1, keepdim=False)

        assert torch.allclose(e, torch.zeros(4, 8))

    def test_positive_energy(self) -> None:
        """Test that energy is always non-negative."""
        x = torch.randn(4, 8, 16)
        e = energy(x, dim=-1, keepdim=False)

        assert torch.all(e >= 0)

    def test_fp16_input(self) -> None:
        """Test energy computation with fp16 input."""
        x = torch.randn(4, 8, 16, dtype=torch.float16)
        e = energy(x, dim=-1, keepdim=False)

        # Should return in original dtype
        assert e.dtype == torch.float16
        assert torch.all(e >= 0)

    def test_fp32_accumulation(self) -> None:
        """Test that fp32 accumulation provides better stability."""
        # Create a case where fp16 might lose precision
        x = torch.full((1000,), 0.1, dtype=torch.float16)
        e = energy(x, dim=-1, keepdim=False)

        # E[x] = mean(x²) = mean(0.01) = 0.01
        expected = torch.tensor(0.01, dtype=torch.float16)
        # With fp32 accumulation, this should be relatively accurate
        assert torch.allclose(e, expected, rtol=0.1)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through energy computation."""
        x = torch.randn(4, 8, 16, requires_grad=True)
        e = energy(x, dim=-1, keepdim=False)
        loss = e.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_scalar_result(self) -> None:
        """Test energy computation resulting in scalar."""
        x = torch.tensor([1.0, 2.0, 3.0])
        e = energy(x, dim=0, keepdim=False)

        assert e.ndim == 0
        expected = torch.tensor(14.0 / 3.0)
        assert torch.allclose(e, expected, atol=1e-6)

    def test_multidimensional(self) -> None:
        """Test energy with 3D tensors (common in transformers)."""
        # [Batch, Sequence, Hidden]
        x = torch.randn(8, 128, 512)
        e = energy(x, dim=-1, keepdim=True)

        assert e.shape == (8, 128, 1)
        assert torch.all(e >= 0)
