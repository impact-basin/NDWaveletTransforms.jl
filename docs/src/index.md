# NDWaveletTransforms.jl

`NDWaveletTransforms` computes discrete wavelet transforms of arrays with
any number of dimensions and a separate number of levels along each axis,
on CPU and GPU.

The transforms are orthogonal and invert to machine precision. An axis of
length `n` supports `l` levels whenever `2^l` divides `n`; it need not be a
power of two.

## Install

```julia
using Pkg
Pkg.add("NDWaveletTransforms")
```

Julia 1.12 or later.

## A first transform

```julia
using NDWaveletTransforms

x = rand(128, 128)
y = dwt(x, WT_D4, 2)      # copy, two levels along each axis
idwt!(y, WT_D4, 2)
x ≈ y                     # true to machine precision
```

`dwt` returns a copy; `dwt!` transforms in place and returns its argument.
The level argument is an `Int` for the same number along every axis, or a
tuple for a separate number per axis:

```julia
dwt(x, WT_D4, (1, 3))     # one level along axis 1, three along axis 2
```

![The same image and its three-level D4 coefficients.](assets/tiling.png)

## The transform families

The prefix and the suffix describe two independent choices. Each family has
a copying and an in-place form, and an inverse.

| | standard | wavelet packet |
|---|---|---|
| separable | `dwt` / `dwt!` | `wpt` / `wpt!` |
| nonstandard | `nsdwt` / `nsdwt!` | `nswpt` / `nswpt!` |

The inverses are `idwt`, `iwpt`, `nsidwt` and `nsiwpt`. A **standard**
transform recurses on the approximation band; a **wavelet packet**
transform recurses on every subband. A **nonstandard** transform applies
every level along one axis before moving to the next, where the standard
form advances one level along every axis in turn; the two agree at a
single level.

The `convention` keyword selects the phase of the detail coefficients; see
[Phase conventions](@ref).

## Where to go next

- [Examples](@ref) for a guided tour.
- [Subbands and views](@ref) for reading and writing individual bands.
- [Bases](@ref) for the built-in filters and your own.
- [API reference](@ref) for the full list.
