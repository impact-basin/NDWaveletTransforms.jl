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
need not be dyadic. The same code runs on a GPU through
KernelAbstractions.jl.

## Documentation

- [Overview](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/)
- [Examples](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/examples/)
- [Phase conventions](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/conventions/)
- [Subbands and views](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/views/)
- [Bases](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/bases/)
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
