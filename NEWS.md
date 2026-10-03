# News

## v1.2.0

### New features

- Phase conventions are now supported via the `convention` keyword.
- `cascade(b, iterations)` calculates scaling/wavelet functions from taps.
- `denoise(x, b, l)` thresholds detail coefficients and inverts. Rules are
  (via `rule` kwarg) are `:universal` (default), `:sure`, or a number; `sigma`
  defaults to the median absolute deviation of the finest detail band.
- `compress(x, b, l; keep)` keeps a fraction of the coefficients and returns
  the reconstruction, the number kept, and the energy retained.
- `threshold!` shrinks coefficients with `:soft`, `:hard` or `:garrote`.
- `keeplargest!` keeps the largest-magnitude coefficients.
- `noisiness` estimates the noise from the median absolute deviation.
- `rtenergy` reports the energy of every first-level subband.
- `sparsity` is Hoyer's sparsity, for comparing bases.

### Bug fixes + performance

- Transforms no longer write out of bounds when given a strided subband
  view from `rtree_view` or `@rtview`.
- Threaded N-D passes no longer box their loop variables, which removes
  an FLoops warning and an allocation from the hot path.

### Internal

- Removed `EllipsisNotation` and `Match` dependencies.
- Added `Statistics` for the median absolute deviation.
