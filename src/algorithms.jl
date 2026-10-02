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

Average `f` over `n` circular shifts of `x`, in place, and return `x`. Each
shift is undone before the results are averaged, so `f` always sees a shifted
copy of the original: this is the Coifman-Donoho cycle-spinning estimator.
The shifts are the primes `prime(start + 1)` upward. Averaging cancels the
edge artefacts that the periodic transform leaves behind, at a cost linear in
`n`.
"""
@fastfun function cyclespinning!(f, x, n=4; start=16)
    n >= 1 || throw(ArgumentError("n must be at least 1"))
    original = copy(x)
    acc = similar(x)
    for i in 1:n
        copyto!(x, original)
        cyclespin!(x, prime(i + start))
        f(x)
        cyclespin!(x, -prime(i + start))
        i == 1 ? copyto!(acc, x) : (acc .+= x)
    end
    x .= acc ./ n
    return x
end

"""
    complement(ψ)

Return the orthogonal complement of the wavelet filter `ψ`, by the
reverse-and-multiply trick.
"""
@fastfun function complement(ψ::SVector{N,T}) :: SVector{N, T} where {N, T}
    SVector{N,T}(ntuple(i -> (-1)^i * ψ[N+1-i], N))
end
