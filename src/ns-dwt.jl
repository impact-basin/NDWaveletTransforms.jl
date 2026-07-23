# dwt.jl -- discrete wavelet transforms, including
# wavelet packet transforms, and their inverses.
# This implementation is provided for speed and
# flexibility; we accept views of signals and
# avoid allocations whenever possible to produce
# rapid transforms. Further, we provide functionality
# for differing transform levels across different
# signal dimensions, critical for analysing data cubes.

# @kernel inbounds=true function _nsdwt1d_kernel!(w, @Const(x), @Const(φ), @Const(ψ))
#     i = @index(Global)
#     l = length(x)
#     n = length(φ)
#     m = l >> 1
#     w[i]   = 0
#     w[i+m] = 0
#     for k=1:n
#         idx = clamp(2i + k-1, 1, l)
#         w[i]   += φ[k] * x[idx]
#         w[i+m] += ψ[k] * x[idx]
#     end
# end
#
# # TODO: tag inbounds.
# @kernel inbounds=true function _insdwt1d_kernel!(w, @Const(x), @Const(φ), @Const(ψ))
#     i    = @index(Global)
#     l = length(x) >> 1
#     n = length(φ)
#     w[i] = 0
#     idx  = 1
#     for k=1:n
#         idx = max((i - k) ÷ 2, 1)
#         w[i] += φ[k] * x[idx] + ψ[k] * x[idx + l]
#     end
# end
#
function _nsdwtinner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{N, T},
) :: Nothing where {T <: Number, N}

    ls = @view w[1:end>>1]
    hs = @view w[(end>>1)+1:end]

    for (outi, sigi) in enumerate(1:2:length(x)-1)
        @inbounds @fastmath @views ls[outi] = dot(b.φ, x[sigi:sigi+b.n-1])
        @inbounds @fastmath @views hs[outi] = dot(b.ψ, x[sigi:sigi+b.n-1])
    end
end

function _insdwtinner_loop!(
    x :: AbstractArray{T,1},
    ls :: SubArray{T, 1},
    hs :: SubArray{T, 1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{N, T},
) :: Nothing where {T <: Number, N}
    @inbounds for (i, j) in enumerate(1:2:length(x)-1)
        # FIXME: clamp!
        @fastmath @views w[j:j+b.n-1] .+=
            ls[i] .* b.φ .+ hs[i] .* b.ψ
    end
end


# Discrete wavelet transform, 1-D
Base.@constprop :aggressive Base.@propagate_inbounds function _dwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    level=1;
    wpt=false
) :: AbstractArray{T,1} where T <: Number

    level <= 0 && return x

    _nsdwtinner_loop!(x, w, b)
    copyto!(x, w)
    if level > 1
        @inbounds _dwt!(@view(x[1:end>>1]), @view(w[1:end>>1]), b, level-1; wpt=wpt)
        @inbounds wpt && _dwt!(@view(x[(end>>1)+1:end]), @view(w[(end>>1)+1:end]), b, level-1; wpt=wpt)
    end

    return x
end

Base.@constprop :aggressive Base.@propagate_inbounds function _idwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    level=1;
    wpt=false
) :: AbstractArray{T,1} where T <: Number

    level <= 0 && return x
    m = length(x) >> 1
    ls = @view x[1:m]
    hs = @view x[m+1:end]

    if level > 1
        _idwt!(
            ls,
            @view(w[1:m]),
            b, level - 1, wpt=wpt
        )
        wpt && _idwt!(
            hs,
            @view(w[m+1:end]),
            b, level - 1, wpt=wpt
        )

    end

    w .= zero(T)

    _insdwtinner_loop!(x, ls, hs, w, b)

    copyto!(x, w)
    return x
end

function nsdwt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}

    if N == 1 
        _dwt!(x, similar(x),
            b, level[1], wpt=wpt
        )
        return x
    end

    s = size(x)[end]

    @floop for i=1:s
        nsdwt!(
            @view(x[.., i]),
            b, level[1:end-1],
            wpt = wpt
        )
    end

    # N.B.: memory copy is good for performance here.
    # 3x speed improvement just by not using views.
    @floop for i in product([1:size(x)[l] for l=1:N-1]...)
        x[i..., :] .= _dwt!(
            x[i..., :],
            Vector{T}(undef, s),
            b, level[end],
            wpt = wpt,
        )
    end
    return x
end

function nsidwt!(
    x :: AbstractArray{T, N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt=false
) :: AbstractArray{T,N} where {T <: Number, N}

    N == 1 && return _idwt!(x,
        Vector{T}(undef, length(x)),
        b, level[1],
        wpt=wpt
    )

    s = size(x)[end]

    # Don't use views here.
    # the memory allocation allows for continuous indexing.
    # This reduces cache misses and improves performance.
    @floop for i in product([1:size(x)[l] for l=1:N-1]...)
         x[i..., :] .= _idwt!(
            x[i..., :],
            Vector{T}(undef, s),
            b, level[end],
            wpt = wpt,
        )
    end

    @threads for i=1:s
        nsidwt!(
            @view(x[.., i]),
            b, level[1:end-1],
            wpt = wpt
        )
    end

    return x
end

#     # => dispatch 1-D transforms over all N-1 dimensional indices.


function nsdwt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l :: Int;
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}
    nsdwt!(x, b, Tuple(l for _ in 1:N), wpt=wpt)
end

function nsidwt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l :: Int;
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}
    nsidwt!(x, b, Tuple(l for _ in 1:N), wpt=wpt)
end

function nsdwt(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l; wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}
    nsdwt!(copy(x), b, l, wpt=wpt)
end

function nsidwt(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l; wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}
    nsidwt!(copy(x), b, l, wpt=wpt)
end

function nswpt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l;
) :: AbstractArray{T,N} where {T <: Number, N}
    nsdwt!(x, b, l, wpt=true)
end

function nsiwpt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l;
) :: AbstractArray{T,N} where {T <: Number, N}
    nsidwt!(x, b, l, wpt=true)
end

function nswpt(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l;
) :: AbstractArray{T,N} where {T <: Number, N}
    nsdwt!(copy(x), b, l, wpt=true)
end

function nsiwpt(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l;
) :: AbstractArray{T,N} where {T <: Number, N}
    nsidwt!(copy(x), b, l, wpt=true)
end
