# NDWaveletTransforms.jl

`NDWaveletTransforms` computes discrete wavelet transforms of arrays with
any number of dimensions and a separate number of levels along each axis. 

An `l`-level transform is supported on an axis of length `n`  whenever `2^l` divides `n`; it need not be a
power of two.

## Example

```julia
using NDWaveletTransforms

x = rand(128, 128)
y = dwt(x, WT_D4, 2)      # copy, two levels along each axis
idwt!(y, WT_D4, 2)
x ≈ y                     # true to machine precision
```

`dwt` returns a copy; `dwt!` transforms in-place and returns its argument.
The level argument is an `Int` for the same number along every axis, or a
tuple for a separate number per axis:

```julia
dwt(x, WT_D4, (1, 3))     # one level along axis 1, three along axis 2
```

![The same image and its three-level D4 coefficients.](assets/tiling.png)

## Transform types

Standard, nonstandard, and packet transforms are supported. Transforms are
indicated by name. A **standard** transform recurses on the approximation
band; a **wavelet packet** transform recurses on every subband. A
**nonstandard** transform applies every level along one axis before moving
to the next, where the standard form advances one level along every axis in
turn; the two agree at a single level.

| | standard | wavelet packet |
|---|---|---|
| separable | `dwt` / `dwt!` | `wpt` / `wpt!` |
| nonstandard | `nsdwt` / `nsdwt!` | `nswpt` / `nswpt!` |

The inverses are `idwt`, `iwpt`, `nsidwt` and `nsiwpt`.

The `convention` keyword selects the phase of the detail coefficients; see
[Phase conventions](@ref).

## Denoising and compression

Coefficients can be shrunk before inverting. [`denoise`](@ref) estimates
noise from the finest detail band, thresholds those coefficients, and
inverts. [`compress`](@ref) instead keeps a fraction of the coefficients.
See [Algorithms](@ref) for details.

```julia
clean = denoise(noisy, WT_D4, 4)
small = compress(image, WT_D4, 3; keep = 0.05)
```

## Transforms on-GPU

Transforms are also implemented for GPU via KernelAbstractions.jl. To perform an
on-GPU transform, pass a GPU array. All flavours of transform are implemented.

```julia
using CUDA
g = CuArray(rand(Float32, 1024, 1024))
y = dwt(g, WT_D4, 3)
```
