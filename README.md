# NDWaveletTransforms.jl

This package implements the discrete wavelet transform, its inverse, and friends - the WPT, and nonstandard flavours of both.

Users will probably prefer the more feature-complete Wavelets.jl library; this package emphasises flexibility over features. Namely, this package supports transforms of arbitrary-dimensional signals, with different transform levels along each dimension. The signal need not be dyadic in any dimension; perfect reconstruction is assured up to an N-level transform in any dimension so long as the length has a factor of 2^N.

This is useful for applications like sparsifying tensors, as well as the analysis of signals with strange dimensions (in my primary use-case, signals are 304 x 16,384 x 85).

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

# Performance

With Haar filters, we're about parity with Wavelets.jl. Other filters are slower; this will be fixed in a future release.

Benchmarked on a Ryzen 9 5950X with 32 threads:

```julia
julia> using Wavelets
julia> import NDWaveletTransforms as NDWT
julia> x = rand(Float32, 2048, 2048);
julia> @benchmark dwt(x, wavelet(WT.haar), 3)
BenchmarkTools.Trial: 60 samples with 1 evaluation per sample.
 Range (min … max):  82.646 ms … 86.174 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     84.350 ms              ┊ GC (median):    0.62%
 Time  (mean ± σ):   84.361 ms ±  1.104 ms  ┊ GC (mean ± σ):  0.64% ± 0.65%

  ▁▁▃                        ▆██▆  ▁                       ▆   
  ███▄▇▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄████▄▄█▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▁▇▇▁▄▄█▄ ▁
  82.6 ms         Histogram: frequency by time        86.1 ms <

 Memory estimate: 16.24 MiB, allocs estimate: 7204.

julia> @benchmark NDWT.dwt(x, NDWT.WT_HAAR, 3)
BenchmarkTools.Trial: 47 samples with 1 evaluation per sample.
 Range (min … max):   79.275 ms … 676.522 ms  ┊ GC (min … max):  0.00% … 88.24%
 Time  (median):      82.693 ms               ┊ GC (median):     0.00%
 Time  (mean ± σ):   107.424 ms ± 116.112 ms  ┊ GC (mean ± σ):  24.14% ± 18.52%

  █                                                              
  █▁▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▁▁▁▁▅ ▁
  79.3 ms       Histogram: log(frequency) by time        677 ms <

 Memory estimate: 48.22 MiB, allocs estimate: 7302.
```

# To-do

* Biorthogonal wavelets
* Performance: filters with > 2 taps have a `mod1()` wrap for periodic boundary implementation. This is slower than it needs to be.
* More performant wt_index() implementation.
* GPU implementation, via KernelAbstractions.jl - at present, this is implemented but not performant.

Contributions are more than welcome :-).

# License & legal

Copyright (c) Henry Eshbaugh <henry.eshbaugh@physics.ox.ac.uk>.

This package is licensed under the terms of the GNU LGPL, v3. See LICENSE.md for text.

No AI was used in the development of this package.
