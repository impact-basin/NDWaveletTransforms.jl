"""
    cyclespin!(x, n)

Circularly shift `x` in place by `n` samples, wrapping the tail to the
front. A matrix shifts both axes by the same amount.
"""
@fastfun cyclespin!(x, n) = x .= @view x[
    mod1.(1+n:end+n, end),
    mod1.(1+n:end+n, end),
]

"""
    cyclespinning!(f, x, n = 4; start = 16)

Apply `f` to `x` under `n` successive circular shifts, undoing each shift
afterwards. The shifts are the primes `prime(start + 1)` upward. Spin
cycling averages out edge artefacts when `f` is a transform or a denoiser.
"""
@fastfun function cyclespinning!(f, x, n=4; start=16)
    for p in (prime(i + start) for i=1:n)
        cyclespin!(x, p)
        f(x)
        cyclespin!(x, -p)
    end
end

"""
    complement(ψ)

Return the orthogonal complement of the wavelet filter `ψ`, by the
reverse-and-multiply trick.
"""
@fastfun function complement(ψ::SVector{N,T}) :: SVector{N, T} where {N, T}
    SVector{N,T}(ntuple(i -> (-1)^i * ψ[N+1-i], N))
end
