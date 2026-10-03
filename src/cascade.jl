# cascade.jl -- the cascade algorithm: filters to scaling and wavelet functions.

# One subdivision: upsample by two, convolve with the filter, scale by √2.
@inline function _subdivide(v::AbstractVector{T}, h::AbstractVector{T}) where {T}
    u = zeros(T, 2length(v))
    u[1:2:end] .= v
    c = zeros(T, 2length(v) + length(h) - 1)
    @inbounds for j in eachindex(h)
        for i in eachindex(u)
            c[i + j - 1] += u[i] * h[j]
        end
    end
    return sqrt(T(2)) .* c
end

"""
    cascade(basis, iterations) -> (φ, ψ)

Sample the scaling function `φ` and the wavelet function `ψ` of `basis`
with the cascade algorithm.

The cascade starts from the box function and applies `iterations`
subdivisions of the refinement equation
`φ(t) = √2 Σ_k h[k] φ(2t - k)`, where `h` is the scaling filter. Each step
doubles the resolution, so the vectors are sampled at
`k / 2^iterations`; the nonzero samples cover the support `[0, N - 1]` of
the filters, where `N` is the number of taps. The wavelet function is one
high-pass subdivision of the previous scaling iterate, so both vectors have
the same length.

`iterations` must be at least 1. Six to eight iterations are enough to plot
most bases.

# Examples

```julia
φ, ψ = cascade(WT_D4, 8)
t = (0:length(φ) - 1) ./ 2^8
```

See the [Algorithms](@ref) and [Bases](@ref) pages.
"""
function cascade(b::WTOrthogonalBasis{N,T}, iterations::Int) where {N,T}
    iterations >= 1 || throw(ArgumentError("iterations must be at least 1"))
    h = collect(T.(b.φ))
    g = collect(T.(b.ψ))
    φ = T[1]
    for _ in 1:iterations
        φ = _subdivide(φ, h)
    end
    # the wavelet applies the high-pass filter on the first refinement,
    # then keeps refining with the scaling filter
    ψ = _subdivide(T[1], g)
    for _ in 1:(iterations - 1)
        ψ = _subdivide(ψ, h)
    end
    return φ, ψ
end
