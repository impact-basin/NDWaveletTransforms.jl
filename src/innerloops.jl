# -----------------
# FORWARD TRANSFORM
# -----------------

# 2-tap transform: no mod1 branching.
@fastfun function _dwt_inner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{2, F},
) :: Nothing where {T <: Number, F <: Number}

    ls = @view w[1:end>>1]
    hs = @view w[(end>>1)+1:end]
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    @inbounds for (outi, sigi) in enumerate(1:2:length(x)-1)
        x1 = x[sigi]
        x2 = x[sigi+1]
        @fastmath ls[outi] = φ[1] * x1 + φ[2] * x2
        @fastmath hs[outi] = ψ[1] * x1 + ψ[2] * x2
    end
    nothing
end

# N-tap transform: mod1 branching only in the wrap-around tail.
@fastfun function _dwt_inner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{N, F},
) :: Nothing where {T <: Number, F <: Number, N}

    ls = @view w[1:end>>1]
    hs = @view w[(end>>1)+1:end]
    n = length(x)
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    nplain = max(0, (n - N + 2) >> 1)

    @inbounds for outi in 1:nplain
        sigi = 2 * outi - 1
        sφ = zero(T)
        sψ = zero(T)
        # @simd wins for short filters but spills for long ones; N is a
        # compile-time constant so the branch is free.
        if N <= 8
            @fastmath @simd for k in 1:N
                xv = x[sigi + k - 1]
                sφ += φ[k] * xv
                sψ += ψ[k] * xv
            end
        else
            @fastmath for k in 1:N
                xv = x[sigi + k - 1]
                sφ += φ[k] * xv
                sψ += ψ[k] * xv
            end
        end
        ls[outi] = sφ
        hs[outi] = sψ
    end

    @inbounds for outi in nplain+1:n>>1
        sigi = 2 * outi - 1
        sφ = zero(T)
        sψ = zero(T)
        @fastmath for k in 1:N
            xv = x[mod1(sigi + k - 1, n)]
            sφ += φ[k] * xv
            sψ += ψ[k] * xv
        end
        ls[outi] = sφ
        hs[outi] = sψ
    end
    nothing
end

# -----------------
# INVERSE TRANSFORM
# -----------------

# 2-tap transform: no mod1 branching.
@fastfun function _idwt_inner_loop!(
    l :: AbstractArray{T, 1},
    h :: AbstractArray{T, 1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{2, F},
) :: Nothing where {T <: Number, F <: Number}
    φ = T.(b.φ)
    ψ = T.(b.ψ)
    @inbounds for (i, j) in enumerate(1:2:length(l)+length(h)-1)
        li = l[i]
        hi = h[i]
        @fastmath w[j]   += li * φ[1] + hi * ψ[1]
        @fastmath w[j+1] += li * φ[2] + hi * ψ[2]
    end
    nothing
end

@fastfun function _idwt_inner_loop!(
    l :: AbstractArray{T, 1},
    h :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis{N, F},
) :: Nothing where {T <: Number, N, F <: Number}
    nw = length(w)
    m = length(l)
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    # Window [j, j+N-1] wraps only when j > nw-N+1
    nplain = max(0, (nw - N + 2) >> 1)

    @inbounds for i in 1:min(m, nplain)
        j = 2 * i - 1
        li = l[i]
        hi = h[i]
        # view broadcast vectorises best for short filters;
        # the scalar @simd loop for long ones.
        if N <= 8
            @fastmath @views w[j:j+N-1] .+= li .* φ .+ hi .* ψ
        else
            @fastmath @simd for k in 1:N
                w[j + k - 1] += li * φ[k] + hi * ψ[k]
            end
        end
    end

    @inbounds for i in min(m, nplain)+1:m
        j = 2 * i - 1
        li = l[i]
        hi = h[i]
        @fastmath for k in 1:N
            w[mod1(j + k - 1, nw)] += li * φ[k] + hi * ψ[k]
        end
    end
    nothing
end

# Single-level write-through passes used by the N-D drivers: the transform
# writes into the second buffer without copying back, and the driver tracks
# which buffer holds the data.
@fastfun function _dwt_level1!(x, w, b)
    _dwt_inner_loop!(x, w, b)
    nothing
end

@fastfun function _idwt_level1!(x, w, b)
    m = length(x) >> 1
    fill!(w, zero(eltype(w)))
    _idwt_inner_loop!(view(x, 1:m), view(x, m+1:length(x)), w, b)
    nothing
end

# Discrete wavelet transform, 1-D
@fastfun function _dwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    level=1;
    wpt=false
) :: AbstractArray{T,1} where T <: Number

    level <= 0 && return x
    n = length(x)

    src, dst = x, w
    nreal = 0
    for lv in 1:level
        bsize = n >> (lv - 1)
        bsize < 2 && break
        nblocks = wpt ? (1 << (lv - 1)) : 1
        for bi in 0:nblocks-1
            r0 = bi * bsize + 1
            _dwt_inner_loop!(
                view(src, r0:r0+bsize-1),
                view(dst, r0:r0+bsize-1),
                b,
            )
        end
        src, dst = dst, src
        nreal += 1
    end

    if wpt
        isodd(nreal) && copyto!(x, w)
    else
        for i in 1:2:nreal-1
            lo = n >> i
            hi = n >> (i - 1)
            copyto!(view(x, lo+1:hi), view(w, lo+1:hi))
        end
        isodd(nreal) && copyto!(view(x, 1:n>>(nreal-1)), view(w, 1:n>>(nreal-1)))
    end

    return x
end

@fastfun function _idwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    level=1;
    wpt=false
) :: AbstractArray{T,1} where T <: Number

    level <= 0 && return x
    n = length(x)
    zt = zero(T)

    src, dst = x, w
    nreal = 0
    for d in 0:level-1
        bsize = n >> (level - 1 - d)
        bsize < 2 && continue
        nblocks = wpt ? (1 << (level - 1 - d)) : 1
        csize = bsize >> 1
        for bi in 0:nblocks-1
            b0 = bi * bsize + 1
            lv = view(src, b0:b0+csize-1)
            if wpt
                hv = view(src, b0+csize:b0+bsize-1)
            else
                hv = view(x, b0+csize:b0+bsize-1)
                if dst === x
                    copyto!(view(w, b0+csize:b0+bsize-1), hv)
                    hv = view(w, b0+csize:b0+bsize-1)
                end
            end
            dv = view(dst, b0:b0+bsize-1)
            fill!(dv, zt)
            _idwt_inner_loop!(lv, hv, dv, b)
        end
        src, dst = dst, src
        nreal += 1
    end

    isodd(nreal) && copyto!(x, w)

    return x
end
