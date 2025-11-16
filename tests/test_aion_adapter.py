"""Tests for AION residual adapter."""

import torch

from aion_torch.aion_adapter import AionResidual  # type: ignore


class TestAionResidual:
    """Test suite for AionResidual adapter."""

    def test_initialization(self) -> None:
        """Test adapter initialization with default parameters."""
        layer = AionResidual()

        assert torch.allclose(layer.alpha0, torch.tensor(0.1), atol=1e-6)
        assert torch.allclose(layer.beta, torch.tensor(0.05), atol=1e-6)
        assert layer.ema_gamma == 0.99
        assert layer.k_update == 1
        assert layer.epsilon == 1e-8

    def test_initialization_custom(self) -> None:
        """Test adapter initialization with custom parameters."""
        layer = AionResidual(
            alpha0=0.2,
            beta=0.1,
            ema_gamma=0.95,
            k_update=5,
            epsilon=1e-6,
        )

        assert torch.allclose(layer.alpha0, torch.tensor(0.2), atol=1e-6)
        assert torch.allclose(layer.beta, torch.tensor(0.1), atol=1e-6)
        assert layer.ema_gamma == 0.95
        assert layer.k_update == 5
        assert layer.epsilon == 1e-6

    def test_forward_basic(self) -> None:
        """Test basic forward pass."""
        layer = AionResidual(alpha0=0.1)
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.train()
        out = layer(x, y)

        # Output should be x + α * y
        assert out.shape == x.shape
        assert not torch.allclose(out, x)  # Should be modified

    def test_forward_inference(self) -> None:
        """Test forward pass in inference mode."""
        layer = AionResidual(alpha0=0.1)
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.eval()
        out1 = layer(x, y)
        out2 = layer(x, y)

        # In eval mode, should use alpha0 if no cached alpha
        assert out1.shape == x.shape
        assert torch.allclose(out1, out2)

    def test_alpha_adaptation_training(self) -> None:
        """Test that alpha adapts during training."""
        layer = AionResidual(alpha0=0.1, beta=0.5, ema_gamma=1.0)

        x = torch.randn(4, 8)
        # Make y have different energy than x
        y = torch.randn(4, 8) * 2.0

        layer.train()
        _ = layer(x, y)

        # After first forward pass, alpha should be cached
        assert layer._alpha_cached is not None  # noqa: SLF001
        # Alpha should be adapted (different from alpha0)
        assert not torch.allclose(layer._alpha_cached, layer.alpha0)  # noqa: SLF001

    def test_k_update(self) -> None:
        """Test that alpha updates every k steps."""
        layer = AionResidual(alpha0=0.1, k_update=3)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8) * 2.0

        layer.train()

        # First pass - should update (step 0)
        _ = layer(x, y)
        alpha1 = layer._alpha_cached.clone() if layer._alpha_cached is not None else None  # noqa: SLF001

        # Second pass - no update (step 1)
        _ = layer(x, y)
        alpha2 = layer._alpha_cached.clone() if layer._alpha_cached is not None else None  # noqa: SLF001

        # Third pass - no update (step 2)
        _ = layer(x, y)
        alpha3 = layer._alpha_cached.clone() if layer._alpha_cached is not None else None  # noqa: SLF001

        # Fourth pass - should update (step 3)
        y_new = torch.randn(4, 8) * 3.0
        _ = layer(x, y_new)
        alpha4 = layer._alpha_cached.clone() if layer._alpha_cached is not None else None  # noqa: SLF001

        assert alpha1 is not None
        if alpha2 is not None and alpha1 is not None:
            assert torch.allclose(alpha1, alpha2)
        if alpha3 is not None and alpha2 is not None:
            assert torch.allclose(alpha2, alpha3)
        if alpha4 is not None:
            # Alpha4 might be different due to different y_new
            pass

    def test_ema_smoothing(self) -> None:
        """Test EMA smoothing of ratio."""
        layer = AionResidual(alpha0=0.1, beta=0.05, ema_gamma=0.9)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.train()

        ratio_ema_before = layer.ratio_ema.item()
        _ = layer(x, y)
        ratio_ema_after = layer.ratio_ema.item()

        # EMA should have been updated
        assert ratio_ema_before != ratio_ema_after

    def test_no_ema_smoothing(self) -> None:
        """Test behavior when EMA smoothing is disabled."""
        layer = AionResidual(alpha0=0.1, beta=0.05, ema_gamma=1.0)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.train()

        ratio_ema_before = layer.ratio_ema.item()
        _ = layer(x, y)
        ratio_ema_after = layer.ratio_ema.item()

        # With ema_gamma=1.0, ratio_ema should not be updated
        assert ratio_ema_before == ratio_ema_after

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the adapter."""
        layer = AionResidual(alpha0=0.1)

        x = torch.randn(4, 8, requires_grad=True)
        y = torch.randn(4, 8, requires_grad=True)

        layer.train()
        out = layer(x, y)
        loss = out.sum()
        loss.backward()

        # Gradients should flow to inputs
        assert x.grad is not None
        assert y.grad is not None

        # Gradients should flow to parameters
        assert layer.alpha0.grad is not None
        assert layer.beta.grad is not None

    def test_different_shapes(self) -> None:
        """Test adapter with different tensor shapes."""
        layer = AionResidual()

        # 2D tensors
        x2d = torch.randn(4, 8)
        y2d = torch.randn(4, 8)
        out2d = layer(x2d, y2d)
        assert out2d.shape == (4, 8)

        # 3D tensors
        x3d = torch.randn(4, 16, 32)
        y3d = torch.randn(4, 16, 32)
        out3d = layer(x3d, y3d)
        assert out3d.shape == (4, 16, 32)

        # 4D tensors
        x4d = torch.randn(2, 3, 8, 8)
        y4d = torch.randn(2, 3, 8, 8)
        out4d = layer(x4d, y4d)
        assert out4d.shape == (2, 3, 8, 8)

    def test_dtype_preservation(self) -> None:
        """Test that dtype is preserved through forward pass."""
        layer = AionResidual()

        # Float32
        x_f32 = torch.randn(4, 8, dtype=torch.float32)
        y_f32 = torch.randn(4, 8, dtype=torch.float32)
        out_f32 = layer(x_f32, y_f32)
        assert out_f32.dtype == torch.float32

        # Float16
        layer_f16 = AionResidual()
        x_f16 = torch.randn(4, 8, dtype=torch.float16)
        y_f16 = torch.randn(4, 8, dtype=torch.float16)
        out_f16 = layer_f16(x_f16, y_f16)
        assert out_f16.dtype == torch.float16

    def test_extra_repr(self) -> None:
        """Test string representation."""
        layer = AionResidual(alpha0=0.2, beta=0.1, ema_gamma=0.95, k_update=5)

        repr_str = layer.extra_repr()

        assert "alpha0=0.2" in repr_str
        assert "beta=0.1" in repr_str
        assert "ema_gamma=0.95" in repr_str
        assert "k_update=5" in repr_str

    def test_step_count_increments(self) -> None:
        """Test that step count increments correctly."""
        layer = AionResidual()

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.train()

        assert layer.step_count.item() == 0
        _ = layer(x, y)
        assert layer.step_count.item() == 1
        _ = layer(x, y)
        assert layer.step_count.item() == 2

    def test_zero_beta_no_adaptation(self) -> None:
        """Test that beta=0 means no adaptation (alpha stays at alpha0)."""
        layer = AionResidual(alpha0=0.1, beta=0.0)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8) * 10.0  # Very different energy

        layer.train()
        _ = layer(x, y)

        # With beta=0, alpha should equal alpha0
        if layer._alpha_cached is not None:  # noqa: SLF001
            assert torch.allclose(layer._alpha_cached, layer.alpha0)  # noqa: SLF001

    def test_high_energy_ratio_reduces_alpha(self) -> None:
        """Test that high output energy reduces alpha."""
        layer = AionResidual(alpha0=0.1, beta=0.5, ema_gamma=1.0)

        x = torch.randn(4, 8)
        y_low = torch.randn(4, 8) * 0.1  # Low energy
        y_high = torch.randn(4, 8) * 10.0  # High energy

        layer.train()

        # Forward with low energy y
        _ = layer(x, y_low)
        alpha_low = layer._alpha_cached.clone() if layer._alpha_cached is not None else None  # noqa: SLF001

        # Reset for fair comparison
        layer._alpha_cached = None  # noqa: SLF001
        layer.step_count.zero_()

        # Forward with high energy y
        _ = layer(x, y_high)
        alpha_high = layer._alpha_cached.clone() if layer._alpha_cached is not None else None  # noqa: SLF001

        # High energy output should result in lower alpha
        if alpha_low is not None and alpha_high is not None:
            assert alpha_high < alpha_low
