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

Views accept `_` as a "do-nothing" index. This is helpful when operating on signals transformed to different levels in different dimensions. For instance,

```julia
x = rand(32, 32)
dwt!(x, WT_HAAR, (1, 2))

@rtview x[:ll, :_l] # 16x8:  deepest scaling band
@rtview x[:_l]      # 32x16: scaling band of axis 2
```

`rtree_views(x)` returns every band at the current level as a tuple.
This is often useful when e.g. plotting transforms.

```julia
ll, lh, hl, hh = rtree_views(x) 
```
