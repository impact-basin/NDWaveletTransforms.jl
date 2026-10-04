# News

## v1.2.1

`rtree_view()` now accepts variadic and symbol-vector arguments for subbands. This indexing recurses into the tree. For instance, `rtree_view(x, :ll, :ll)`, a synonym for `rtree_view(x, [:ll, :ll])`, returns a view of the LL subband of the LL subband of `x`. This is often useful when working with many transform levels and the WPT.
