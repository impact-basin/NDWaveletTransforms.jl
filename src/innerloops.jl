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
    x :: AbstractArray{T,1},
    ls :: SubArray{T, 1},
    hs :: SubArray{T, 1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{2, F},
) :: Nothing where {T <: Number, F <: Number}
    for (i, j) in enumerate(1:2:length(x)-1)
        @fastmath @views w[j:j+1] .+=
            ls[i] .* T.(b.φ) .+ hs[i] .* T.(b.ψ)
    end
end

# N-tap transform: mod1 branching.
@turbofun function _idwt_inner_loop!(
    x :: AbstractArray{T,1},
    ls :: SubArray{T, 1},
    hs :: SubArray{T, 1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{N, F},
) :: Nothing where {T <: Number, N, F <: Number}
    for (i, j) in enumerate(1:2:length(x)-1)
        @fastmath @views w[mod1.(j:j+N-1, end)] .+=
            ls[i] .* T.(b.φ) .+ hs[i] .* T.(b.ψ)
    end
end
