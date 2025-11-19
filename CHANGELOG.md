# Changelog

All notable changes to AION-Torch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- No unreleased changes yet

---

## [0.2.0] - 2024-11-19

**This is a major quality and stability release with breaking changes.**

### ⚠️ Breaking Changes
- **Removed `k_update` parameter**: The `k_update` parameter has been removed from `AionResidual.__init__()`. Alpha now updates every forward pass in training mode to ensure correct behavior in distributed training (DataParallel/DDP). This fixes a critical bug where replicas would have inconsistent alpha values in multi-GPU training.

  **Migration guide**:
  ```python
  # Before (broken in distributed training)
  layer = AionResidual(alpha0=0.1, beta=0.05, k_update=4)

  # After (works in all training modes)
  layer = AionResidual(alpha0=0.1, beta=0.05)

  # For performance: use gradient accumulation instead
  accumulation_steps = 4
  for i, batch in enumerate(dataloader):
      loss = model(batch) / accumulation_steps
      loss.backward()
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```

### Added
- Comprehensive input validation with detailed error messages
  - Validates `alpha0 > 0`
  - Validates `beta >= 0`
  - Validates `0 <= ema_gamma <= 1`
  - Validates `epsilon > 0`
  - Validates tensor shapes match in forward pass
  - Validates tensors are non-empty
- 28 new test cases covering edge cases and numerical stability
  - Device placement tests (CUDA)
  - dtype support tests (float16, bfloat16, float32)
  - Numerical stability tests (NaN, Inf, zero energy, extreme values)
  - Gradient flow tests
  - State dict save/load tests
  - Eval mode consistency tests
  - Scalar ratio handling tests

### Fixed
- **Critical: State dict handling**: `alpha_cached` is now properly registered as a buffer, ensuring it's saved and loaded with model checkpoints. Previously, `alpha_cached` was a private attribute that wasn't persisted.
- **Critical: Distributed training support**: Removed `k_update` to ensure consistent alpha values across all replicas in DataParallel and DistributedDataParallel training.
- Step count now only increments in training mode (previously incremented in eval mode)
- Scalar ratio handling: properly handles edge cases where energy ratio is a scalar tensor
- Empty tensor validation: checks for empty tensors before shape validation for better error messages

### Changed
- Alpha updates every forward pass in training mode (previously controlled by `k_update`)
- `extra_repr()` output no longer includes `k_update`
- Documentation updated to recommend gradient accumulation instead of `k_update` for performance optimization

### 📊 Performance Notes
- Overhead remains ~36% per training step (unoptimized baseline)
- Use gradient accumulation to amortize overhead across multiple batches
- Engineering optimizations (operation fusion, lower precision) can reduce overhead to ~5%

### 🔧 Technical Details
- **Total test coverage**: 77 tests (76 passing, 1 skipped)
- **Lines of code changed**: ~500 lines across 5 files
- **New validation rules**: 8 parameter and input validations
- **Bug fixes**: 2 critical, 3 moderate

### 📝 Migration Impact
- **Low impact for single-GPU users**: Just remove `k_update` parameter
- **High impact for multi-GPU users**: Fixes critical distributed training bug
- **No impact on model weights**: Existing checkpoints compatible (after loading)
- **Recommended action**: Update to v0.2.0 immediately for distributed training

---

## [0.1.0] - 2024-01-15

### Added
- Initial alpha release of AION-Torch
- `AionResidual` layer for adaptive residual scaling
- Energy computation with fp32 accumulation
- EMA smoothing for ratio stability
- Learnable `alpha0` and `beta` parameters
- Registry system for pluggable adapters
- Comprehensive test suite
- Benchmark results for 600-layer transformers
- MIT license

### ⚠️ Known Issues (All Fixed in v0.2.0)
- ❌ `k_update` parameter causes inconsistent behavior in distributed training → **Fixed**
- ❌ `alpha_cached` not properly saved in state dict → **Fixed**
- ❌ Step count increments in eval mode → **Fixed**
- ❌ Missing input validation → **Fixed**
- ❌ Poor error messages for invalid inputs → **Fixed**
- ❌ No tests for edge cases (NaN, Inf, empty tensors) → **Fixed**

### 📦 Release Notes
- **Status**: Alpha release
- **Stability**: Experimental
- **Production ready**: No (use v0.2.0 instead)
- **Upgrade path**: Direct upgrade to v0.2.0 recommended

---

## Version History

- **v0.2.0** (2024-11-19): Major stability release, distributed training support
- **v0.1.0** (2024-01-15): Initial alpha release

## Links

[Unreleased]: https://github.com/Croxus-Labs/aion-torch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Croxus-Labs/aion-torch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Croxus-Labs/aion-torch/releases/tag/v0.1.0
