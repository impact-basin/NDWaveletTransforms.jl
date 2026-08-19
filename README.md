# NDWaveletTransforms.jl

This package implements the discrete wavelet transform, its inverse, and friends - the WPT, and nonstandard flavours of both.

Users will probably prefer the more feature-complete Wavelets.jl library; this package is tailored to a particular application of DWTs. Namely, this package supports transforms of arbitrary-dimensional signals, with different transform levels along each dimension. This is useful for applications like sparsifying high-rank tensors, as well as the analysis of signals with strange dimensions (in my primary use-case, signals are 304 x 16,384 x 85).

# Usage

The functions `dwt()`, `idwt()`, `wpt()`, and `iwpt()` are supported. The argument order is the input array, the wavelet basis, and the transform level. The `dwt!()` family mutates the input array in-place. Nonstandard transforms are prefixed with "ns", e.g. `nsdwt()` or `nsdwt!()`.

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
## Supported Wavelets

Orthogonal only, at the moment.

| Kind | Symbol |
| ---- | ------ |
| Haar | `WT_HAAR` |
| Daubechies, N vanishing moments, to N=20 | `WT_DN`, e.g. `WT_D2` |
| Coiflets | `WT_COIF2`, `WT_COIF4`, ..., `WT_COIF10` |
| Symlets | `WT_SYM4`, `WT_SYM5`, ..., `WT_SYM10` |
| Battle-Lemarie | `WT_BATTLE4`, `WT_BATTLE6` |
| Beyl | `WT_BEYL` |
| Vaidyanathan | `WT_VAID` |

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
@wtview x[:ll, ll] # => LL subband of LL subband of x
```

# To-do

* Biorthogonal wavelets
* Performance: filters with > 2 taps have a lazy `mod1()` wrap for periodic boundary implementation. This is slower than it needs to be.
* GPU implementation, via KernelAbstractions.jl - at present, this is implemented but not performant.

# License & legal

Copyright (c) Henry Eshbaugh <henry.eshbaugh@physics.ox.ac.uk>.

This package is licensed under the terms of the GNU LGPL, v3. See LICENSE.md for text.


No AI was used in the development of this package.
