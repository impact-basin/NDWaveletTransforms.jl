# NDWaveletTransforms.jl

Discrete wavelet transforms of arrays with any number of dimensions, with a
separate number of levels along each axis, on CPU and GPU.

```julia
using NDWaveletTransforms

x = rand(128, 128)
y = dwt(x, WT_D4, 2)
idwt!(y, WT_D4, 2)
x ≈ y   # true
```

The transforms are orthogonal and invert to machine precision. A dimension
of length `n` supports `l` levels whenever `2^l` divides `n`, so the signal
need not be dyadic.

The same code runs on a GPU through KernelAbstractions.jl. Pass a GPU array
and the dispatch follows; every transform family is covered.

Coefficients can be shrunk before inverting, which denoises or compresses:

```julia
clean = denoise(noisy, WT_D4, 4)                # threshold the detail bands
small = compress(image, WT_D4, 3; keep = 0.05)  # keep 5% of the coefficients
```

## Documentation

- [Overview](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/)
- [Examples](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/examples/)
- [Phase conventions](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/conventions/)
- [Subbands and views](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/views/)
- [Bases](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/bases/)
- [Algorithms](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/algorithms/)
- [API reference](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/api/)

## Install

```julia
using Pkg
Pkg.add("NDWaveletTransforms")
```

Julia 1.12 or later.

## Contributing

Contributions are welcome.

## License

GNU LGPL v3. See [LICENSE.md](LICENSE.md).
