"""Tests for AION residual adapter."""

import pytest
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
        assert layer.epsilon == 1e-8

    def test_initialization_custom(self) -> None:
        """Test adapter initialization with custom parameters."""
        layer = AionResidual(
            alpha0=0.2,
            beta=0.1,
            ema_gamma=0.95,
            epsilon=1e-6,
        )

        assert torch.allclose(layer.alpha0, torch.tensor(0.2), atol=1e-6)
        assert torch.allclose(layer.beta, torch.tensor(0.1), atol=1e-6)
        assert layer.ema_gamma == 0.95
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
        assert layer.alpha_cached is not None
        # Alpha should be adapted (different from alpha0)
        assert not torch.allclose(layer.alpha_cached, layer.alpha0)

    def test_alpha_updates_every_step(self) -> None:
        """Test that alpha updates every training step."""
        layer = AionResidual(alpha0=0.1, beta=0.05, ema_gamma=1.0)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8) * 2.0

        layer.train()

        # First pass
        _ = layer(x, y)
        alpha1 = layer.alpha_cached.clone()

        # Second pass with different input - alpha should change
        y_new = torch.randn(4, 8) * 3.0
        _ = layer(x, y_new)
        alpha2 = layer.alpha_cached.clone()

        # Alpha should be different because inputs changed
        assert not torch.allclose(alpha1, alpha2)
        assert layer.step_count.item() == 2

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
        layer = AionResidual(alpha0=0.2, beta=0.1, ema_gamma=0.95)

        repr_str = layer.extra_repr()

        assert "alpha0=0.2" in repr_str
        assert "beta=0.1" in repr_str
        assert "ema_gamma=0.95" in repr_str

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
        assert torch.allclose(layer.alpha_cached, layer.alpha0)

    def test_high_energy_ratio_reduces_alpha(self) -> None:
        """Test that high output energy reduces alpha."""
        layer = AionResidual(alpha0=0.1, beta=0.5, ema_gamma=1.0)

        x = torch.randn(4, 8)
        y_low = torch.randn(4, 8) * 0.1  # Low energy
        y_high = torch.randn(4, 8) * 10.0  # High energy

        layer.train()

        # Forward with low energy y
        _ = layer(x, y_low)
        alpha_low = layer.alpha_cached.clone()

        # Reset for fair comparison
        layer.alpha_cached.copy_(layer.alpha0)
        layer.step_count.zero_()

        # Forward with high energy y
        _ = layer(x, y_high)
        alpha_high = layer.alpha_cached.clone()

        # High energy output should result in lower alpha
        assert alpha_high < alpha_low

    def test_input_validation_alpha0_negative(self) -> None:
        """Test that negative alpha0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha0 must be positive"):
            AionResidual(alpha0=-0.1)

    def test_input_validation_alpha0_zero(self) -> None:
        """Test that zero alpha0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha0 must be positive"):
            AionResidual(alpha0=0.0)

    def test_input_validation_beta_negative(self) -> None:
        """Test that negative beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be non-negative"):
            AionResidual(beta=-0.1)

    def test_input_validation_ema_gamma_too_low(self) -> None:
        """Test that ema_gamma < 0 raises ValueError."""
        with pytest.raises(ValueError, match="ema_gamma must be in"):
            AionResidual(ema_gamma=-0.1)

    def test_input_validation_ema_gamma_too_high(self) -> None:
        """Test that ema_gamma > 1 raises ValueError."""
        with pytest.raises(ValueError, match="ema_gamma must be in"):
            AionResidual(ema_gamma=1.1)

    def test_input_validation_epsilon_zero(self) -> None:
        """Test that zero epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            AionResidual(epsilon=0.0)

    def test_input_validation_epsilon_negative(self) -> None:
        """Test that negative epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            AionResidual(epsilon=-1e-8)

    def test_forward_shape_mismatch(self) -> None:
        """Test that shape mismatch in forward raises ValueError."""
        layer = AionResidual()
        x = torch.randn(4, 8)
        y = torch.randn(4, 9)  # Different shape

        with pytest.raises(ValueError, match="Input shapes must match"):
            layer(x, y)

    def test_forward_empty_tensor_x(self) -> None:
        """Test that empty tensor x raises ValueError."""
        layer = AionResidual()
        x = torch.empty(0, 8)
        y = torch.randn(4, 8)

        with pytest.raises(ValueError, match="cannot be empty"):
            layer(x, y)

    def test_forward_empty_tensor_y(self) -> None:
        """Test that empty tensor y raises ValueError."""
        layer = AionResidual()
        x = torch.randn(4, 8)
        y = torch.empty(4, 0)

        with pytest.raises(ValueError, match="cannot be empty"):
            layer(x, y)

    def test_state_dict_save_load(self) -> None:
        """Test that state dict properly saves and loads alpha_cached."""
        layer1 = AionResidual(alpha0=0.2, beta=0.1)
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        # Train layer1 to update alpha_cached
        layer1.train()
        _ = layer1(x, y)
        alpha_before = layer1.alpha_cached.clone()

        # Save state dict
        state_dict = layer1.state_dict()

        # Create new layer and load state
        layer2 = AionResidual(alpha0=0.1, beta=0.05)
        layer2.load_state_dict(state_dict)

        # Verify alpha_cached was loaded
        assert torch.allclose(layer2.alpha_cached, alpha_before)
        assert torch.allclose(layer2.alpha0, layer1.alpha0)
        assert torch.allclose(layer2.beta, layer1.beta)
        assert torch.allclose(layer2.ratio_ema, layer1.ratio_ema)
        assert layer2.step_count.item() == layer1.step_count.item()

    def test_state_dict_preserves_alpha_cached(self) -> None:
        """Test that alpha_cached is preserved across state dict operations."""
        layer = AionResidual(alpha0=0.1, beta=0.05)
        x = torch.randn(4, 8)
        y = torch.randn(4, 8) * 2.0

        layer.train()
        _ = layer(x, y)
        original_alpha = layer.alpha_cached.clone()

        # Save and reload
        state_dict = layer.state_dict()
        layer.load_state_dict(state_dict)

        # Alpha should be preserved
        assert torch.allclose(layer.alpha_cached, original_alpha)

    def test_step_count_not_incremented_in_eval(self) -> None:
        """Test that step_count doesn't increment in eval mode."""
        layer = AionResidual()
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.eval()
        initial_step = layer.step_count.item()

        # Multiple forward passes in eval mode
        _ = layer(x, y)
        _ = layer(x, y)
        _ = layer(x, y)

        # Step count should remain unchanged in eval mode
        assert layer.step_count.item() == initial_step

    def test_scalar_ratio_handling(self) -> None:
        """Test that scalar ratios are handled correctly."""
        layer = AionResidual(alpha0=0.1, beta=0.05, ema_gamma=1.0)

        # Create inputs that result in scalar energy (single element)
        x = torch.randn(1, 1)
        y = torch.randn(1, 1)

        layer.train()
        out = layer(x, y)

        # Should work without errors
        assert out.shape == x.shape
        assert layer.alpha_cached is not None

    def test_nan_input_handling(self) -> None:
        """Test behavior with NaN inputs."""
        layer = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        y[0, 0] = float("nan")

        layer.train()
        out = layer(x, y)

        # Should propagate NaN (or handle gracefully)
        # The layer should not crash, but NaN may propagate
        assert out.shape == x.shape

    def test_inf_input_handling(self) -> None:
        """Test behavior with Inf inputs."""
        layer = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        y[0, 0] = float("inf")

        layer.train()
        out = layer(x, y)

        # Should handle Inf (may propagate or be handled by epsilon)
        assert out.shape == x.shape

    def test_zero_energy_tensors(self) -> None:
        """Test behavior with zero energy tensors."""
        layer = AionResidual(alpha0=0.1, beta=0.05, epsilon=1e-8)

        # Zero tensor
        x = torch.zeros(4, 8)
        y = torch.zeros(4, 8)

        layer.train()
        out = layer(x, y)

        # Should handle zero energy gracefully
        assert out.shape == x.shape
        # Output should be zero (x + alpha * y where both are zero)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_very_small_energy(self) -> None:
        """Test numerical stability with very small energy values."""
        layer = AionResidual(alpha0=0.1, beta=0.05, epsilon=1e-8)

        # Very small values
        x = torch.randn(4, 8) * 1e-10
        y = torch.randn(4, 8) * 1e-10

        layer.train()
        out = layer(x, y)

        # Should handle without numerical issues
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_very_large_energy(self) -> None:
        """Test numerical stability with very large energy values."""
        layer = AionResidual(alpha0=0.1, beta=0.05, epsilon=1e-8)

        # Very large values
        x = torch.randn(4, 8) * 1e10
        y = torch.randn(4, 8) * 1e10

        layer.train()
        out = layer(x, y)

        # Should handle without overflow
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_device_placement(self) -> None:
        """Test that buffers are on the correct device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        layer = AionResidual(alpha0=0.1, beta=0.05)
        layer = layer.cuda()

        x = torch.randn(4, 8, device="cuda")
        y = torch.randn(4, 8, device="cuda")

        layer.train()
        out = layer(x, y)

        # All buffers should be on CUDA
        assert layer.alpha0.device.type == "cuda"
        assert layer.beta.device.type == "cuda"
        assert layer.alpha_cached.device.type == "cuda"
        assert layer.ratio_ema.device.type == "cuda"
        assert layer.step_count.device.type == "cuda"
        assert out.device.type == "cuda"

    def test_bfloat16_dtype(self) -> None:
        """Test with bfloat16 dtype."""
        if not hasattr(torch, "bfloat16"):
            pytest.skip("bfloat16 not available")

        layer = AionResidual()
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        y = torch.randn(4, 8, dtype=torch.bfloat16)

        layer.train()
        out = layer(x, y)

        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape

    def test_single_sample_batch(self) -> None:
        """Test with batch size of 1."""
        layer = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(1, 8)
        y = torch.randn(1, 8)

        layer.train()
        out = layer(x, y)

        assert out.shape == (1, 8)
        assert layer.alpha_cached is not None

    def test_large_batch(self) -> None:
        """Test with large batch size."""
        layer = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(128, 8)
        y = torch.randn(128, 8)

        layer.train()
        out = layer(x, y)

        assert out.shape == (128, 8)
        assert layer.alpha_cached is not None

    def test_multiple_backward_passes(self) -> None:
        """Test gradient accumulation with multiple backward passes."""
        layer = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(4, 8, requires_grad=True)
        y = torch.randn(4, 8, requires_grad=True)

        layer.train()

        # First backward pass
        out1 = layer(x, y)
        loss1 = out1.sum()
        loss1.backward(retain_graph=True)

        # Verify gradients exist after first backward
        assert x.grad is not None
        assert y.grad is not None
        assert layer.alpha0.grad is not None
        assert layer.beta.grad is not None

        # Zero gradients
        x.grad.zero_()
        y.grad.zero_()
        if layer.alpha0.grad is not None:
            layer.alpha0.grad.zero_()
        if layer.beta.grad is not None:
            layer.beta.grad.zero_()

        # Second backward pass
        out2 = layer(x, y)
        loss2 = out2.sum()
        loss2.backward()

        # Gradients should accumulate correctly
        assert x.grad is not None
        assert y.grad is not None
        assert layer.alpha0.grad is not None
        assert layer.beta.grad is not None

    def test_eval_mode_consistency(self) -> None:
        """Test that eval mode produces consistent outputs."""
        layer = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.eval()

        # Multiple forward passes should produce same output
        out1 = layer(x, y)
        out2 = layer(x, y)
        out3 = layer(x, y)

        assert torch.allclose(out1, out2)
        assert torch.allclose(out2, out3)

    def test_state_dict_partial_load(self) -> None:
        """Test loading state dict with strict=False."""
        layer1 = AionResidual(alpha0=0.2, beta=0.1)
        layer2 = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer1.train()
        _ = layer1(x, y)

        # Create a partial state dict (missing some keys)
        state_dict = layer1.state_dict()
        # Remove one key to test partial loading
        del state_dict["beta"]

        # Should work with strict=False
        layer2.load_state_dict(state_dict, strict=False)

        # Alpha0 should be loaded
        assert torch.allclose(layer2.alpha0, layer1.alpha0)
        # Beta should remain unchanged
        assert torch.allclose(layer2.beta, torch.tensor(0.05))

    def test_alpha_cached_initialization(self) -> None:
        """Test that alpha_cached is initialized correctly."""
        layer = AionResidual(alpha0=0.2, beta=0.1)

        # alpha_cached should be initialized to alpha0
        assert torch.allclose(layer.alpha_cached, layer.alpha0)

    def test_ratio_ema_initialization(self) -> None:
        """Test that ratio_ema is initialized correctly."""
        layer = AionResidual()

        # ratio_ema should be initialized to 1.0
        assert torch.allclose(layer.ratio_ema, torch.tensor(1.0))

    def test_step_count_initialization(self) -> None:
        """Test that step_count is initialized correctly."""
        layer = AionResidual()

        # step_count should be initialized to 0
        assert layer.step_count.item() == 0

    def test_very_small_epsilon(self) -> None:
        """Test with very small epsilon value."""
        layer = AionResidual(alpha0=0.1, beta=0.05, epsilon=1e-12)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.train()
        out = layer(x, y)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_continuous_alpha_updates(self) -> None:
        """Test that alpha continues to update over many steps."""
        layer = AionResidual(alpha0=0.1, beta=0.05, ema_gamma=0.9)

        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer.train()

        # Run many forward passes
        for _ in range(100):
            _ = layer(x, y)

        # Alpha should have been updated
        assert layer.alpha_cached is not None
        assert torch.isfinite(layer.alpha_cached)
        assert layer.step_count.item() == 100

    def test_ema_gamma_boundary_values(self) -> None:
        """Test EMA gamma at boundary values."""
        # Test ema_gamma = 0.0 (full replacement)
        layer0 = AionResidual(alpha0=0.1, beta=0.05, ema_gamma=0.0)
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        layer0.train()
        _ = layer0(x, y)

        # With ema_gamma=0, ratio_ema should be completely replaced
        assert layer0.ratio_ema is not None

        # Test ema_gamma = 1.0 (no update)
        layer1 = AionResidual(alpha0=0.1, beta=0.05, ema_gamma=1.0)
        initial_ratio = layer1.ratio_ema.clone()

        layer1.train()
        _ = layer1(x, y)

        # With ema_gamma=1.0, ratio_ema should not change
        assert torch.allclose(layer1.ratio_ema, initial_ratio)

    def test_5d_tensors(self) -> None:
        """Test with 5D tensors (edge case for shape handling)."""
        layer = AionResidual()

        x = torch.randn(2, 3, 4, 5, 6)
        y = torch.randn(2, 3, 4, 5, 6)

        out = layer(x, y)

        assert out.shape == (2, 3, 4, 5, 6)

    def test_parameter_gradients_exist(self) -> None:
        """Test that learnable parameters receive gradients."""
        layer = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(4, 8, requires_grad=True)
        y = torch.randn(4, 8, requires_grad=True)

        layer.train()
        out = layer(x, y)
        loss = out.sum()
        loss.backward()

        # Both parameters should have gradients
        assert layer.alpha0.grad is not None
        assert layer.beta.grad is not None
        assert torch.isfinite(layer.alpha0.grad)
        assert torch.isfinite(layer.beta.grad)

    def test_alpha_cached_gradient_flow(self) -> None:
        """Test that gradients flow through alpha_cached."""
        layer = AionResidual(alpha0=0.1, beta=0.05)

        x = torch.randn(4, 8, requires_grad=True)
        y = torch.randn(4, 8, requires_grad=True)

        layer.train()
        # Do one forward pass to update alpha_cached
        _ = layer(x, y)

        # Now do another forward pass
        out = layer(x, y)
        loss = out.sum()
        loss.backward()

        # Gradients should flow through alpha_cached to alpha0 and beta
        assert layer.alpha0.grad is not None
        assert layer.beta.grad is not None
