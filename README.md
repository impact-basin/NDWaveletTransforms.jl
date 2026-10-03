# NDWaveletTransforms.jl

This package implements the discrete wavelet transform, its inverse, and friends. See the [documentation](https://impact-basin.github.io/NDWaveletTransforms.jl/stable/) for details.

Users will probably prefer the more feature-complete Wavelets.jl library; this package emphasises flexibility over features. Namely, this package supports transforms of arbitrary-dimensional signals, with different transform levels along each dimension. The signal need not be dyadic in any dimension; perfect reconstruction is assured up to an N-level transform in any dimension so long as the length has a factor of 2^N.

This is useful for applications like sparsifying tensors, as well as the analysis of signals with strange dimensions (in my primary use-case, signals are 304 x 16,384 x 85).

# Usage

The functions dwt(), idwt(), wpt(), and iwpt() are supported. The argument order is the input array, the wavelet basis, and the transform level. The dwt!() family mutates the input array in-place. Nonstandard transforms are prefixed with "ns", e.g. nsdwt() or nsdwt!().

Standard forward/inverse transform:

```julia
x = rand(128,128)
y = dwt(x, WT_HAAR, 1)
idwt!(y, WT_HAAR, 1)
x ≈ y # => true
```

```julia
x = rand(128,128)
wpt(x, WT_HAAR, 1) ≈ dwt(x, WT_HAAR, 1; wpt=true) # => true
```

As mentioned earlier, the level count can differ per-dimension:

```julia
dwt(x, WT_D4, (1, 3))     # level-1 along dim 1, level-3 along dim 2
```

Convenience macros give subband views into the transform:

```julia
@rtview y[:ll] .= 0       # zero the approximation band
```

Shrinkage denoising and compression are supported:

```julia
clean = denoise(noisy, WT_D4, 4)                # threshold the detail bands
small = compress(image, WT_D4, 3; keep = 0.05)  # keep 5% of the coefficients
```

A GPU array dispatches to GPU kernels:

```julia
using CUDA
y = dwt(CuArray(rand(Float32, 1024, 1024)), WT_D4, 3)



## License

GNU LGPL v3. See [LICENSE.md](LICENSE.md).
