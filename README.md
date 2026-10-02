# NDWaveletTransforms.jl

Discrete wavelet transforms of arrays with any number of dimensions, with a
separate number of levels along each axis, on CPU and GPU.

```julia
using NDWaveletTransforms

x = rand(128, 128)
y = dwt(x, WT_D4, 2)      # two levels along each axis
idwt!(y, WT_D4, 2)
x ≈ y                     # true to machine precision
```

The transforms are orthogonal and invert exactly. An axis of length `n`
supports `l` levels whenever `2^l` divides `n`, so lengths need not be dyadic.

## Examples

The level count can differ per axis:

```julia
dwt(x, WT_D4, (1, 3))     # one level along axis 1, three along axis 2
```

Subbands are views into the coefficient array:

```julia
@rtview y[:ll] .= 0       # zero the approximation band
```

Coefficients can be shrunk before inverting, which denoises or compresses:

```julia
clean = denoise(noisy, WT_D4, 4)                # threshold the detail bands
small = compress(image, WT_D4, 3; keep = 0.05)  # keep 5% of the coefficients
```

A GPU array dispatches to the GPU kernels. Nothing else changes:

```julia
using CUDA
y = dwt(CuArray(rand(Float32, 1024, 1024)), WT_D4, 3)
```

## Install

```julia
using Pkg
Pkg.add("NDWaveletTransforms")
```

Julia 1.12 or later.

## Documentation

The [documentation](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/)
covers the four transform families, the phase conventions, the subband views,
denoising and compression, and the API.

## License

GNU LGPL v3. See [LICENSE.md](LICENSE.md).
