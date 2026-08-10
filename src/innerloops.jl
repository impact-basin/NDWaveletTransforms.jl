# -----------------
# FORWARD TRANSFORM
# -----------------

# 2-tap transform: no mod1 branching.
@turbofun function _dwt_inner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{2, F},
) :: Nothing where {T <: Number, F <: Number}

    ls = @view w[1:end>>1]
    hs = @view w[(end>>1)+1:end]

    for (outi, sigi) in enumerate(1:2:length(x)-1)
        @fastmath @views ls[outi] = dot(T.(b.φ), x[sigi:sigi+1])
        @fastmath @views hs[outi] = dot(T.(b.ψ), x[sigi:sigi+1])
    end
end

# N-tap transform: mod1 branching.
@turbofun function _dwt_inner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{N, F},
) :: Nothing where {T <: Number, F <: Number, N}

    ls = @view w[1:end>>1]
    hs = @view w[(end>>1)+1:end]

    for (outi, sigi) in enumerate(1:2:length(x)-1)
        @fastmath @views ls[outi] = dot(T.(b.φ), x[mod1.(sigi:sigi+N-1, end)])
        @fastmath @views hs[outi] = dot(T.(b.ψ), x[mod1.(sigi:sigi+N-1, end)])
    end
end

# -----------------
# INVERSE TRANSFORM
# -----------------

# 2-tap transform: no mod1 branching.
@turbofun function _idwt_inner_loop!(
    l :: AbstractArray{T, 1},
    h :: AbstractArray{T, 1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{2, F},
) :: Nothing where {T <: Number, F <: Number}
    for (i, j) in enumerate(1:2:length(l)+length(h)-1)
        @fastmath @views w[j:j+1] .+=
            l[i] .* T.(b.φ) .+ h[i] .* T.(b.ψ)
    end
end

# N-tap transform: mod1 branching.
# TODO: faster mod1!
@turbofun function _idwt_inner_loop!(
    l :: AbstractArray{T, 1},
    h :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis{N, F},
) :: Nothing where {T <: Number, N, F <: Number}
    for (i, j) in enumerate(1:2:length(l)+length(h)-1)
        @fastmath @views w[mod1.(j:j+N-1, end)] .+=
            l[i] .* T.(b.φ) .+ h[i] .* T.(b.ψ)
    end
end

# Discrete wavelet transform, 1-D
@turbofun function _dwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    level=1;
    wpt=false
) :: AbstractArray{T,1} where T <: Number

    level <= 0 && return x

    _dwt_inner_loop!(x, w, b)
    # TODO: remove copyto!
    copyto!(x, w)
    if level > 1
        @strided _dwt!(
            x[1:end>>1],
            w[1:end>>1],
            b, level-1;
            wpt=wpt,
        )
        @strided wpt && _dwt!(
            x[(end>>1)+1:end],
            w[(end>>1)+1:end],
            b, level-1;
            wpt=wpt,
        )
    end

    return x
end

@turbofun function _idwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    level=1;
    wpt=false
) :: AbstractArray{T,1} where T <: Number

    level <= 0 && return x
    m = length(x) >> 1

    if level > 1
        @strided _idwt!(
            x[1:m],
            w[1:m],
            b, level - 1, wpt=wpt
        )
        @strided wpt && _idwt!(
            x[m+1:end],
            w[m+1:end],
            b, level - 1, wpt=wpt
        )
    end

    w .= zero(T)

    @strided _idwt_inner_loop!(
        x[1:m],
        x[m+1:end],
        w, b,
    )

    copyto!(x, w)
    return x
end
