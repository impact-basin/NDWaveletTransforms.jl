# News

## v1.2.1

- `rtree_view()` now accepts variadic and symbol-vector arguments for subbands. This indexing recurses into the tree. For instance, `rtree_view(x, :ll, :ll)`, a synonym for `rtree_view(x, [:ll, :ll])`, returns a view of the LL subband of the LL subband of `x`. This is often useful when working with many transform levels and the WPT.

## v1.2.2

- Aliased `rtview()`, `rtviews()` to `rtree_view()` and `rtree_views()`, respectively.
- Fixed a regression where views returned from `rtview()` and friends would return views of
  `StridedArray`s, which, by loss of type information, would prevent recursive `dwt()` invocations.
- Added `PrecompileTools.jl` dep. 1D, 2D, and 3D forward/inverse transforms are now precompiled
  for `Float32` and `Float64` arguments under both phase conventions.
- Added a docstring for the toplevel module.
- Fixed a bug where `@rtview` was processing its arguments too aggressively; expressions like `@rtview x[:ll] .= 2y` now work as expected.
