# Subbands and views

Transformed arrays are r-trees (generalised quadtrees) of subbands.
`rtree_view` returns in-place views of subbands.

```julia
x = rand(256, 256)
dwt!(x, WT_D4, 3)

ll1 = rtree_view(x, :ll)         # 128x128: 1-level LL band
lh1 = rtree_view(x, :lh)         # 128x128: 1-level LH band
```

The `@rtview` macro indexes into the signal with `rtree_view()` for convenience.
It is often useful when masking subbands or performing other filtering.

```julia
ll3 = @rtview x[:ll, :ll, :ll]   #  32x32:  3-level LL band
@rtview x[:ll] .= 0              # zero the 1-level LL band
@rtview x[:hh] .*= 0.5           # half the 1-level HH band
```

`rtree_views(x)` returns every band at the current level as a tuple.
This is often useful when e.g. plotting transforms.

```julia
ll, lh, hl, hh = rtree_views(x) 
```
