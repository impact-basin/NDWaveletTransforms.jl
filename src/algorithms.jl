# TODO implement cascade.
#
# function cascade(φ, ψ; n = 4, l=4)
# 	if ψ == nothing
# 		ψ = copy(φ) |> reverse
# 		ψ[2:2:end] .*= -1
# 	end
# 	len = n
# 	ret = ones(n)
# 	for i ∈ 1:l-1
# 		ret = conv(φ, ret)
# 		ret = ret |> upsample
# 		len *= 2
# 	end
# 	return conv(φ, ret), conv(ψ, ret)
# 	ret[1:length(ret)÷2]
# end
#
# # cascade algorithm for generating continuous-time wavelets.
# function cascade(wt::WTOrthogonalBasis; n=10)
#
#     # initialise working memory
#     ϕ = copy(wt.φ)
#
#     # iterated convolution.
#     for i=1:n-1
#         ϕ = conv(ϕ, wt.φ)
#         ϕ = up2(ϕ)
#     end
#
#     # final convolution round for scaling/wavelet functions
#     φ = up2(ϕ)
#     return (conv(φ, wt.φ), conv(φ, wt.ψ))
# end

# function scalogram(s :: SIGNAL_1D,
#                    b :: WTOrthogonalBasis = D1;
#                    l :: Int = wtlevels(length(s))+1)
#
#     d = dwt(s, b, l)
#     n = length(s)
#     ret = Matrix{typeof(s[1])}(l, n >> 1)
#     for i=1:l
#         a, b = n >> i, n >> (i-1)
#         @views ret[i,1:2^l:end] .= d[a:b]
#     end
#     return ret
# end

# TODO implement thresholding, denoising, etc.

# TODOiimplement basis substitution
# we want to take the projection of s onto ψ1 and
# subtract it off - taking it out of this subspace.
# we then substitute the lost signal energy by adding on
# another basis, ψ2, with the same projection coefficient.
#
# This is basically Gram-Schmidt orthogonalisation for one basis element,
# with the addition of a new basis element after.
#
# this is equivalent to calculating the wavelet coefficient:
#   c = sum(s .* ψ1)
#
# then, we subtract this off from the original signal:
#   s .-= c .* ψ1
#
# finally, we add the new signal back on:
#   s .+= c .* ψ2
#
# this gives
#   s = s + c * ψ2 - c * ψ1
#     = s + c * (ψ2 - ψ1)
#
#  and we have our weird expression.

# TODO iterate over the array properly
function substitute_basis(s, ψ1, ψ2)
    return s .+ sum(s .* ψ1) .* (ψ2 - ψ1)
end


function substitute_basis!(s, ψ1, ψ2)
    s = substitute_basis(s, ψ1, ψ2)
end

# TODO iterate over the array properly
function null_basis!(s, ψ)
    s .-= sum(s .* ψ) .* ψ
end

function null_basis(s, ψ)
    sc = copy(s)
    null_basis!(sc, ψ)
    sc
end

# currying for fun and profit.
substitute_basis(ψ1,  ψ2) = s -> substitute_basis(s, ψ1, ψ2)
substitute_basis!(ψ1, ψ2) = s -> substitute_basis!(s, ψ1, ψ2)

# now, how do we generate the weird bases we might like?
# easy: kronecker products.

function kronbasis(ψs::Vector)
    length(ψs) == 0 && error("fuck you!")
    length(ψs) == 1 && return ψs[1]
    ret = ψs[1]
    for ψ in ψs[2:end]
        ret = kron(ret, ψ)
    end
    return ret
end

cyclespin!(x, n) =
    x .= @view x[
        mod1.(1+n:end+n, end),
        mod1.(1+n:end+n, end),
    ]

function cyclespinning!(f, x, n=4; start=16)
    for p in (mprime(i + start) for i=1:4)
        cyclespin!(x, p) 
        f(x)
        cyclespin!(x, -p)
    end
end

function sifting!(
    f :: Function,
    x :: AbstractArray{T, N},
    n :: Int,
    start = 32,
) :: Nothing where {T <: Number, N}
    w = similar(x)
    for i=start:start+n 
        p = mprime(mtprime(i))
        circshift!(w, x, (p, p))
        f(w)
        circshift!(x, w, (-p, -p))
    end
    nothing
end

@memoize mprime(i) = prime(i)

"""
    complement(ψ)

    Find the orthogonal complement of the wavelet filter ψ.
    This is by using the classic "reverse-and-mutliply" trick.
"""
function complement(ψ::SVector{N,T}) :: SVector{N, T} where {N, T}
    SVector{N,T}(ntuple(i -> (-1)^i * ψ[N+1-i], N))
end

labs(x) = log10.(abs.(x) .+ eps())

macro turbofun(expr)
    return quote
        Base.@constprop :aggressive Base.@propagate_inbounds $expr
    end |> esc
end
