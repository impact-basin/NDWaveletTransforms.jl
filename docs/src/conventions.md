# Phase conventions

The scaling coefficients are the same in both conventions. Only the detail
coefficients differ, by a cyclic rotation.

Every transform takes a `convention` keyword:

- `:aligned` (the default) applies the scaling filter φ and the wavelet
  filter ψ to the same input window. This is the textbook phase, and
  matches PyWavelets and MATLAB-style implementations.
- `:wavelets` reproduces the coefficient layout of Wavelets.jl. The ψ
  window starts `ntaps - 2` taps before the φ window, so each detail band
  is the aligned one rotated by `(ntaps - 2) / 2`.

For a two-tap filter the two coincide. For longer filters they describe
the same transform with the detail coefficients shifted.

```julia
x = rand(1024)
a = dwt(x, WT_D4, 3; convention = :aligned)    # default
b = dwt(x, WT_D4, 3; convention = :wavelets)   # matches Wavelets.jl

idwt!(a, WT_D4, 3; convention = :aligned)      # round trips
```

`convention` is dispatched at compile time, so neither choice costs
anything at run time. Use `:aligned` unless you need to match Wavelets.jl
or reproduce a coefficient layout computed elsewhere.

## One dimension

A rectangular pulse, with its level-1 detail coefficients under both
conventions. The pulse has two edges, so the detail band has two responses.
Each response is the same shape in both conventions, rotated by three
coefficients.

![The level-1 detail band of a one-dimensional transform under both conventions.](assets/phase-1d.png)

## Two dimensions

The same comparison on a compact feature, with a four-level `WT_D8`
transform. The first panel is the image. The middle and right panels are
the same coefficients under the two conventions, with the subband
boundaries drawn. The approximation band in the top-left corner is
identical, because the scaling coefficients agree. Every detail band is
rotated, and the rotation is a larger fraction of a coarser band, so the
displacement grows with depth.

![The image, and the same multilevel coefficients under the two conventions.](assets/phase-2d.png)

## The wavelet under each convention

The filters are the same, so [`cascade`](@ref) produces the same φ and ψ for
both conventions. What differs is the phase at which the wavelet is placed
relative to the coefficient index. The `:wavelets` detail coefficients are
the `:aligned` ones cyclically shifted by `(ntaps - 2) / 2` positions, while
the scaling coefficients are identical.

```julia
φ, ψ = cascade(WT_D4, 8)     # the same for both conventions
ntaps = length(WT_D4.φ)      # 8
δ = (ntaps - 2) / 2          # 3 coefficient positions
```

![The scaling and wavelet functions, and the detail coefficients of an impulse, under both conventions.](assets/phase-wavelets.png)

The scaling function is unchanged. The two copies of ψ are the same curve,
offset by `δ`; the impulse response confirms it, with the two detail patterns
of identical shape separated by three coefficients. Matching another library
is a matter of relabelling the detail bands, not of choosing a different
wavelet.
