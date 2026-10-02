# Algorithms

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
