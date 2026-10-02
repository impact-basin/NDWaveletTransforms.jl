# Subbands and views

A transformed array is a mosaic of subbands. `rtree_view` returns a view of
one of them, so reading or writing the view reads or writes the coefficient
array.

```julia
x = rand(256, 256)
dwt!(x, WT_D4, 3)

ll1 = rtree_view(x, :ll)         # 128x128: the level-1 approximation band
lh1 = rtree_view(x, :lh)         # 128x128: low along axis 1, high along axis 2
ll3 = @rtview x[:ll, :ll, :ll]   #  32x32: the coarsest approximation band
```

A band name is one letter per axis, `l` for low and `h` for high. `rtree_view`
splits the array once, so a name describes the current level rather than the
whole tree. Chaining walks inward.

The `@rtview` macro rewrites its indexing argument into `rtree_view` calls,
which makes it usable on the left of an assignment:

```julia
@rtview x[:ll] .= 0              # zero the approximation band
@rtview x[:hh] .*= 0.5           # scale the finest HH band
```

`rtree_views(x)` returns every band at the current level as a tuple in the
order `1:2^ndims(x)`, with the all-low band first.

## Filtering in the wavelet domain

The result is a `SubArray`, so a transform function accepts it directly and
a mask over the bands selects what survives reconstruction. Keeping one
oriented detail band is a directional filter:

```julia
img = rand(256, 256)
c = dwt(img, WT_D4, 1)

mask = trues(size(c))
rtree_view(mask, :lh) .= false    # drop one orientation
rtree_view(mask, :hh) .= false
c[.!mask] .= 0
filtered = idwt(c, WT_D4, 1)
```

![Keeping different subbands of the same image.](assets/views-filter.png)

The test image is vertically striped in its upper half and horizontally
striped in its lower half. Keeping the approximation alone blurs both.
Adding `:lh`, which is high along axis 2, restores the vertical stripes and
leaves the horizontal ones blurred; adding `:hl` does the reverse.

Transforming a subband in place transforms that part of the original array:

```julia
x = rand(256, 256)
dwt!(x, WT_D4, 1)                  # one level everywhere
dwt!(@rtview(x[:ll]), WT_D4, 2)    # two more levels on the LL band
idwt!(@rtview(x[:ll]), WT_D4, 2)   # and back
```
