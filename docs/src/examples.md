# Examples

## A one-dimensional signal

A four-level D4 transform splits a signal into one approximation band and
four detail bands. Thresholding the detail bands removes noise while
keeping the part of the signal that is smooth.

```julia
using NDWaveletTransforms
using Random

Random.seed!(1)
t = range(0, 1, length = 2048)
s = sin.(2pi .* 6 .* t) .+ 0.35 .* sin.(2pi .* 45 .* t) .+ 0.05 .* randn(2048)

c = dwt(s, WT_D4, 4)

cut = copy(c)
cut[abs.(cut) .< 0.1] .= 0     # drop the small coefficients
s_denoised = idwt(cut, WT_D4, 4)
```

![A signal and its four-level coefficients.](assets/coefficients.png)

## Sparsity

Smooth signals keep most of their energy in a few wavelet coefficients, and
the rest decay quickly. Keeping the largest coefficients and zeroing the
rest reconstructs the signal to a relative error that falls fast with the
fraction kept. Longer filters do better on smooth signals; Haar does worse,
and pays for it with a blocky reconstruction.

![Relative error against the fraction of retained coefficients.](assets/compression.png)

## An image

```julia
img = rand(512, 512)
c = dwt(img, WT_D4, 3)

mags = sort(abs.(c); rev = true)
k = round(Int, 0.05 * length(c))
compressed = copy(c)
compressed[abs.(compressed) .< mags[k]] .= 0
img_compressed = idwt(compressed, WT_D4, 3)
```

The coefficient array is a mosaic: a coarse approximation in the top-left
corner, and progressively finer detail bands around it.

![The subband tiling of a three-level transform.](assets/tiling.png)

## Levels along each axis

The level argument can be a tuple, one entry per axis, so a rectangular
array can be transformed more deeply along its long axis than its short
one, or left alone along an axis that is already smooth.

```julia
img = rand(256, 256)
dwt(img, WT_D4, (1, 3))    # one level along axis 1, three along axis 2
dwt(img, WT_D4, (3, 1))    # the other way round
```

![The same image at levels (1, 3) and (3, 1).](assets/levels.png)

## Wavelet packets

`wpt` applies the transform to every subband rather than only the
approximation band. The coefficient count is unchanged; the recursion
reaches further into the frequency range, which suits signals whose energy
is not concentrated at low frequency.

```julia
x = rand(256)
c = wpt(x, WT_D4, 3)
iwpt!(c, WT_D4, 3)
c ≈ x
```

## Nonstandard transforms

The `ns` family applies every level along one axis before moving to the
next, where the standard transform advances one level along every axis in
turn. The two agree at a single level and diverge once the transform
recurses.

```julia
img = rand(128, 128)
dwt(img, WT_D4, (1, 1)) ≈ nsdwt(img, WT_D4, (1, 1))   # true
dwt(img, WT_D4, (2, 2)) ≈ nsdwt(img, WT_D4, (2, 2))   # false
```
