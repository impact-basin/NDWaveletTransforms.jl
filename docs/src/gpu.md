# GPU

The transforms run on GPUs through KernelAbstractions.jl. Pass a GPU array
and the dispatch follows.

```julia
using CUDA
using NDWaveletTransforms

x = rand(Float32, 1024, 1024)
g = CuArray(x)

y = dwt(g, WT_D4, 3)     # ran on the GPU
idwt!(y, WT_D4, 3)
Array(y) ≈ x
```

Every family is covered: `dwt`/`idwt`, `wpt`/`iwpt`, `nsdwt`/`nsidwt` and
`nswpt`/`nsiwpt`, forward and inverse, for both phase conventions.

The GPU kernels are for large arrays, where the launch overhead is small
beside the transform. On an RTX 4070 SUPER a 1024x1024 three-level D4 round
trip runs about 54 times faster than the single-threaded CPU path.
