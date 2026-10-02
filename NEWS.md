# News

## v1.2.0

### New features

- Every transform takes a `convention` keyword. `:aligned` (the default) is
  the textbook phase and matches PyWavelets; `:wavelets` reproduces the
  Wavelets.jl coefficient layout. The scaling coefficients are identical
  under both.
- `cascade(b, iterations)` returns the scaling and wavelet functions of an
  orthogonal basis.
- `denoise(x, b, l)` thresholds the detail coefficients and inverts. The
  `rule` is `:universal` (the default), `:sure`, or a number; `sigma`
  defaults to the median absolute deviation of the finest detail band.
- `compress(x, b, l; keep)` keeps a fraction of the coefficients and returns
  the reconstruction, the number kept and the energy retained.
- `threshold!` shrinks coefficients with `:soft`, `:hard` or `:garrote`.
- `keeplargest!` keeps the largest-magnitude coefficients.
- `noisiness` estimates the noise from the median absolute deviation.
- `rtenergy` reports the energy of every first-level subband.
- `sparsity` is Hoyer's sparsity, for comparing bases.

### Bug fixes

- `dwt!` and the other in-place transforms no longer write out of bounds
  when given a strided subband view from `rtree_view` or `@rtview`.
- The threaded N-D passes no longer box their loop variables, which removes
  an FLoops warning and an allocation from the hot path.
- `denoise` thresholds only the detail bands, so the approximation and the
  smooth part of the signal are kept.

### Documentation

- A Documenter site with examples, a phase-convention page, a basis gallery
  and an algorithms page.

### Internal

- Removed the unused `EllipsisNotation` and `Match` dependencies.
- Split `dwt.jl` and `innerloops.jl`, and removed dead helpers.
- Added `Statistics` for the median absolute deviation.
