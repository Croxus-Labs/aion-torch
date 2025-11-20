# Changelog

All notable changes to AION-Torch will be documented in this file.

## [Unreleased]

## [0.3.1] - 2025-11-21

### Changed
- Updated README with documentation links to theoretical papers
- Improved documentation section with clearer organization
- Updated community messaging

## [0.3.0] - 2025-11-19

### Added
- **New `AionBlock` Component**: Added high-level `AionBlock` class in `adapters.py` that implements the Pre-LayerNorm pattern with AION residual scaling.
- **Enhanced Monitoring**: Added `return_stats` parameter to `AionResidual.forward()` to return internal statistics (alpha, ratio) for debugging and visualization.
- **Benchmarking Suite**: Added comprehensive benchmarking scripts in `examples/` for CPU/GPU overhead and stability testing.
- **Visualization Tools**: Added `examples/visualize_results.py` to plot benchmark comparisons.
- Added `tests/test_aion_block.py` for `AionBlock` verification.

### Fixed
- **Distributed Training**: Added distributed synchronization (all_reduce) for energy ratio calculation to ensure consistent alpha values across all ranks in DDP/DataParallel training.
- **Code Quality**: Fixed Pylint warning for unnecessary ellipsis in `ResidualAdapter` protocol definition.

### Performance
- **Optimization**: JIT-compiled `energy()` function using `@torch.jit.script` for faster execution.

---

## [0.2.0] - 2025-11-19

Major stability release with breaking changes for distributed training support.

### Breaking Changes
- Removed `k_update` parameter from `AionResidual.__init__()`
- Alpha now updates every forward pass in training mode
- Migration: Remove `k_update` parameter, use gradient accumulation for performance

```python
# Before
layer = AionResidual(alpha0=0.1, beta=0.05, k_update=4)

# After
layer = AionResidual(alpha0=0.1, beta=0.05)
```

### Added
- Input validation for all parameters (alpha0, beta, ema_gamma, epsilon)
- Tensor shape and empty tensor validation
- 28 new tests for edge cases and numerical stability

### Fixed
- Critical: `alpha_cached` now properly saved in state dict
- Critical: Distributed training support (DataParallel/DDP)
- Step count only increments in training mode
- Scalar ratio edge case handling

### Performance
- Training overhead: ~36% per step (unoptimized baseline)
- Use gradient accumulation to reduce overhead
- Optimizations can reduce overhead to ~5%

---

## [0.1.0] - 2025-11-17

Initial alpha release.

### Added
- `AionResidual` layer for adaptive residual scaling
- Energy computation with fp32 accumulation
- EMA smoothing for ratio stability
- Learnable alpha0 and beta parameters
- Registry system for pluggable adapters
- Comprehensive test suite
- MIT license

### Known Issues
Fixed in v0.2.0:
- `k_update` parameter causes issues in distributed training
- `alpha_cached` not saved in state dict
- Step count increments in eval mode
- Missing input validation

---

[Unreleased]: https://github.com/Croxus-Labs/aion-torch/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/Croxus-Labs/aion-torch/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Croxus-Labs/aion-torch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Croxus-Labs/aion-torch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Croxus-Labs/aion-torch/releases/tag/v0.1.0
