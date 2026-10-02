# Algorithms

## The cascade algorithm

A basis stores two filters. The refinement equation turns them into a
scaling function φ and a wavelet function ψ:

φ(t) = √2 Σ_k h[k] φ(2t - k),    ψ(t) = √2 Σ_k g[k] φ(2t - k),

with `h` the scaling filter and `g` the wavelet filter. `cascade(basis, n)`
samples both by starting from the box function and applying the subdivision
`n` times; each step doubles the resolution.

```julia
using NDWaveletTransforms

φ, ψ = cascade(WT_D4, 8)
t = (0:length(φ) - 1) ./ 2^8     # samples at k / 2^8
```

The scaling function refines with the scaling filter at every step. The
wavelet applies the high-pass filter on the first refinement, then keeps
refining with the scaling filter. The two operators do not commute, so the
order matters.

![The D4 scaling function after n cascade iterations.](assets/cascade-convergence.png)

The iterates converge to the scaling function: the first few are visibly
piecewise linear, and by six or eight the curve is smooth to the eye. The
values at the integers converge more slowly, because a Daubechies scaling
function is only Hölder continuous; low-order bases need more iterations to
match the textbook integer values than to look right.

## Cycle spinning

The transform treats the array as periodic, so a feature near the boundary
wraps around and leaves pseudo-Gibbs oscillations in the coefficients.
Cycle spinning suppresses them: apply the denoiser under several circular
shifts, and undo each shift afterwards. The artefact lands in a different
place each time and averages out; the signal does not.

`cyclespinning!(f, x, n = 4; start = 16)` runs the loop. `f` transforms `x`
in place, like `dwt!`.

```julia
function denoise!(x)
    dwt!(x, WT_D4, 4)
    x[abs.(x) .< 0.1 * maximum(abs.(x))] .= 0
    idwt!(x, WT_D4, 4)
    return x
end

plain = denoise!(copy(noisy))
spun = copy(noisy)
cyclespinning!(denoise!, spun, 8)
```

The shifts are the primes from `prime(start + 1)` upward, so consecutive
calls use different displacements. More shifts reduce the artefact further,
at linear cost.

![Hard thresholding of a noisy step, with and without cycle spinning.](assets/spinning.png)

The thresholded signal overshoots at the step and at the boundary, where
the periodic wrap-around lives. The spin-cycled result follows the step
more closely and rings less.

`cyclespin!(x, n)` performs one circular shift by `n`, in place.
