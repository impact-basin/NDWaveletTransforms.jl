# NDWaveletTransforms.jl

This package implements the discrete wavelet transform, its inverse, and friends - the WPT, and nonstandard flavours of both.

Users will probably prefer the more feature-complete Wavelets.jl library; this package emphasises flexibility over features. Namely, this package supports transforms of arbitrary-dimensional signals, with different transform levels along each dimension. The signal need not be dyadic in any dimension; perfect reconstruction is assured up to an N-level transform in any dimension so long as the length has a factor of 2^N.

This is useful for applications like sparsifying tensors, as well as the analysis of signals with strange dimensions (in my primary use-case, signals are 304 x 16,384 x 85).

# Usage

The functions `dwt()`, `idwt()`, `wpt()`, and `iwpt()` are supported. The argument order is the input array, the wavelet basis, and the transform level. The `dwt!()` family mutates the input array in-place. Nonstandard transforms are prefixed with "ns", e.g. `nsdwt()` or `nsdwt!()`.

## Phase conventions

The transform supports two phase conventions for the wavelet (detail)
coefficients, selected with the `convention` keyword (available on every
transform, forward and inverse, CPU and GPU):

* `convention = :aligned` (default) -- the scaling and wavelet filters act on
  the same input window. This is the textbook phase used by e.g. PyWavelets
  and MATLAB-style implementations.
* `convention = :wavelets` -- reproduces the coefficient layout of
  [Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl): the wavelet
  filter's window starts `N-2` taps before the scaling filter's window, so
  each detail band is the aligned one cyclically shifted by `(N-2)/2`
  positions. For 2-tap (Haar) filters the two conventions coincide.

The scaling coefficients are identical in both conventions, and both
conventions are orthogonal (perfect-reconstruction) transforms. The
convention is dispatched at compile time, so there is no runtime cost for
either choice:

```julia
x = rand(128, 128)
dwt(x, WT_D2, 2; convention = :aligned)   # textbook phase (default)
dwt(x, WT_D2, 2; convention = :wavelets)  # matches Wavelets.jl's dwt(x, w, 2)
```

Standard forward/inverse transform:
```julia
x = rand(128,128)
y = dwt(x, WT_HAAR, 1)
idwt!(y, WT_HAAR, 1)
x ≈ y # => true
```

Wavelet packet transform:

```julia
x = rand(128,128)
wpt(x, WT_HAAR, 1) ≈ dwt(x, WT_HAAR, 1; wpt=true) # => true
```

## GPU support

All transforms (`dwt`/`idwt`, `wpt`/`iwpt`, `nsdwt`/`nsidwt`, `nswpt`/`nsiwpt`)
also run on GPUs via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
Users should pass a GPU array to these functions to trigger the KernelAbstractions.jl backend.

```julia
using CUDA
x = rand(128, 128)
y = dwt(CuArray(x), WT_D4, 3)   # GPU forward
idwt!(y, WT_D4, 3)
x ≈ Array(y)                    # => true
```

On an RTX 4070 SUPER the 1024x1024 3-level D4 round trip is ~54x faster than
the CPU path on one thread.

## Supported Wavelets

Orthogonal only, at the moment.

| Kind | Symbol |
| ---- | ------ |
| Haar | `WT_HAAR` |
| Daubechies, N vanishing moments | `WT_D1`, `WT_D2`, ..., `WT_D20` |
| Coiflets | `WT_COIF2`, `WT_COIF4`, ..., `WT_COIF10` |
| Symlets | `WT_SYM2`, `WT_SYM3`, ..., `WT_SYM10` |
| Battle-Lemarie | `WT_BATTLE2`, `WT_BATTLE4`, `WT_BATTLE6` |
| Beyl | `WT_BEYL` |
| Vaidyanathan | `WT_VAID` |
| Morris minimum-bandwidth | `WT_MB42`, `WT_MB82`, `WT_MB83`, `WT_MB84`, `WT_MB103`, `WT_MB123`, `WT_MB143`, `WT_MB163`, `WT_MB183`, `WT_MB243`, `WT_MB323` |
| Fejer-Korovkin | `WT_FK4`, `WT_FK6`, `WT_FK8`, `WT_FK14`, `WT_FK18`, `WT_FK22` |
| Best-localised Daubechies | `WT_BL7`, `WT_BL9`, `WT_BL10` |
| Han | `WT_HAN23`, `WT_HAN33`, `WT_HAN45`, `WT_HAN55` |

## Constructing your own

Pass a static array to the `WTOrthogonalBasis` constructor. Either the scaling or the wavelet taps can be specified. Normalisation is handled in the constructor. For instance, two instances of the Haar basis:

```julia
const MY_WT_HAAR = WTOrthogonalBasis(ψ = SA{Float64}[1, -1])
const MY_WT_D1 = WTOrthogonalBasis(φ = SA{Float64}[0.7071067811865475, 0.7071067811865475])
```

# Other Tricks

The `@wtview` macro allows for access to subspaces easily, e.g.

```julia
@wtview x[:ll] # => LL subband of x
@wtview x[:ll, :ll] # => LL subband of LL subband of x
```

# To-do

* Biorthogonal wavelets

Contributions are more than welcome :-).

# License & legal

Copyright (c) Henry Eshbaugh <henry.eshbaugh@physics.ox.ac.uk>.

This package is licensed under the terms of the GNU LGPL, v3. See LICENSE.md for text.

No AI was used in the development of this package.
