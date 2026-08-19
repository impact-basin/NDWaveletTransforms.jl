@fastfun cyclespin!(x, n) = x .= @view x[
    mod1.(1+n:end+n, end),
    mod1.(1+n:end+n, end),
]

@fastfun function cyclespinning!(f, x, n=4; start=16)
    for p in (prime(i + start) for i=1:4)
        cyclespin!(x, p) 
        f(x)
        cyclespin!(x, -p)
    end
end

"""
    complement(ψ)

    Find the orthogonal complement of the wavelet filter ψ.
    This is by using the classic "reverse-and-mutliply" trick.
"""
@fastfun function complement(ψ::SVector{N,T}) :: SVector{N, T} where {N, T}
    SVector{N,T}(ntuple(i -> (-1)^i * ψ[N+1-i], N))
end
