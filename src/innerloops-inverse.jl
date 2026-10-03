# -----------------
# INVERSE TRANSFORM
# -----------------

# 2-tap transform: no mod1 branching.
@fastfun function _idwt_inner_loop!(
    l :: AbstractArray{T, 1},
    h :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis{2, F},
    ::Val{:aligned},
) :: Nothing where {T <: Number, F <: Number}

    pl = _strided1(l)
    ph = _strided1(h)
    pw = _strided1(w)
    if pl !== nothing && ph !== nothing && pw !== nothing
        if pl[2] == 1 && ph[2] == 1 && pw[2] == 1
                                              return _idwt_inner_strided!(pl[1], ph[1], pw[1], 1, 1, 1, length(w), b, Val(:aligned), Val(true))
                                          else
                                              return _idwt_inner_strided!(pl[1], ph[1], pw[1], pl[2], ph[2], pw[2], length(w), b, Val(:aligned), Val(false))
                                          end
    end

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

# 2-tap: both conventions coincide.
@fastfun function _idwt_inner_loop!(
    l :: AbstractArray{T, 1},
    h :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis{2, F},
    ::Val{:wavelets},
) :: Nothing where {T <: Number, F <: Number}
    _idwt_inner_loop!(l, h, w, b, Val(:aligned))
    nothing
end

@fastfun function _idwt_inner_strided!(
    pl :: Ptr{T},
    ph :: Ptr{T},
    pw :: Ptr{T},
    sl :: Int,
    sh :: Int,
    sw :: Int,
    nw :: Int,
    b :: WTOrthogonalBasis{2, F},
    ::Val{:aligned},
    ::Val{UNIT},
) :: Nothing where {T <: Number, F <: Number, UNIT}
    m = nw >> 1
    φ = T.(b.φ)
    ψ = T.(b.ψ)
    if UNIT
        @inbounds for i in 1:m
            li = unsafe_load(pl, i)
            hi = unsafe_load(ph, i)
            j = 2 * i - 1
            @fastmath unsafe_store!(pw, unsafe_load(pw, j) + li * φ[1] + hi * ψ[1], j)
            @fastmath unsafe_store!(pw, unsafe_load(pw, j + 1) + li * φ[2] + hi * ψ[2], j + 1)
        end
    else
        @inbounds for i in 1:m
            li = unsafe_load(pl, (i - 1) * sl + 1)
            hi = unsafe_load(ph, (i - 1) * sh + 1)
            j = 2 * i - 1
            @fastmath unsafe_store!(pw, unsafe_load(pw, (j - 1) * sw + 1) + li * φ[1] + hi * ψ[1], (j - 1) * sw + 1)
            @fastmath unsafe_store!(pw, unsafe_load(pw, j * sw + 1) + li * φ[2] + hi * ψ[2], j * sw + 1)
        end
    end
    nothing
end

@fastfun function _idwt_inner_loop!(
    l :: AbstractArray{T, 1},
    h :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis{N, F},
    ::Val{:aligned},
) :: Nothing where {T <: Number, N, F <: Number}

    pl = _strided1(l)
    ph = _strided1(h)
    pw = _strided1(w)
    if pl !== nothing && ph !== nothing && pw !== nothing
        if pl[2] == 1 && ph[2] == 1 && pw[2] == 1
                                              return _idwt_inner_strided!(pl[1], ph[1], pw[1], 1, 1, 1, length(w), b, Val(:aligned), Val(true))
                                          else
                                              return _idwt_inner_strided!(pl[1], ph[1], pw[1], pl[2], ph[2], pw[2], length(w), b, Val(:aligned), Val(false))
                                          end
    end

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

@fastfun function _idwt_inner_strided!(
    pl :: Ptr{T},
    ph :: Ptr{T},
    pw :: Ptr{T},
    sl :: Int,
    sh :: Int,
    sw :: Int,
    nw :: Int,
    b :: WTOrthogonalBasis{N, F},
    ::Val{:aligned},
    ::Val{UNIT},
) :: Nothing where {T <: Number, N, F <: Number, UNIT}
    m = nw >> 1
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    # Window [j, j+N-1] wraps only when j > nw-N+1
    nplain = max(0, (nw - N + 2) >> 1)

    if UNIT
        @inbounds for i in 1:min(m, nplain)
            j = 2 * i - 1
            li = unsafe_load(pl, i)
            hi = unsafe_load(ph, i)
            if N <= 8
                @fastmath @simd for k in 1:N
                    off = j + k - 1
                    unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k] + hi * ψ[k], off)
                end
            else
                @fastmath for k in 1:N
                    off = j + k - 1
                    unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k] + hi * ψ[k], off)
                end
            end
        end

        @inbounds for i in min(m, nplain)+1:m
            j = 2 * i - 1
            li = unsafe_load(pl, i)
            hi = unsafe_load(ph, i)
            @fastmath for k in 1:N
                off = mod1(j + k - 1, nw)
                unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k] + hi * ψ[k], off)
            end
        end
    else
        @inbounds for i in 1:min(m, nplain)
            j = 2 * i - 1
            li = unsafe_load(pl, (i - 1) * sl + 1)
            hi = unsafe_load(ph, (i - 1) * sh + 1)
            if N <= 8
                @fastmath @simd for k in 1:N
                    off = (j + k - 2) * sw + 1
                    unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k] + hi * ψ[k], off)
                end
            else
                @fastmath for k in 1:N
                    off = (j + k - 2) * sw + 1
                    unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k] + hi * ψ[k], off)
                end
            end
        end

        @inbounds for i in min(m, nplain)+1:m
            j = 2 * i - 1
            li = unsafe_load(pl, (i - 1) * sl + 1)
            hi = unsafe_load(ph, (i - 1) * sh + 1)
            @fastmath for k in 1:N
                off = (mod1(j + k - 1, nw) - 1) * sw + 1
                unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k] + hi * ψ[k], off)
            end
        end
    end
    nothing
end

# Wavelets.jl-compatible phase: the scaling coefficients feed the aligned
# window [2i-1, 2i+N-2] (tail wraps) while the wavelet coefficients feed the
# window [2i-N+1, 2i] (head wraps). The two passes accumulate into the same
# (pre-zeroed) output, so their order does not matter.
@fastfun function _idwt_inner_loop!(
    l :: AbstractArray{T, 1},
    h :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis{N, F},
    ::Val{:wavelets},
) :: Nothing where {T <: Number, N, F <: Number}

    pl = _strided1(l)
    ph = _strided1(h)
    pw = _strided1(w)
    if pl !== nothing && ph !== nothing && pw !== nothing
        if pl[2] == 1 && ph[2] == 1 && pw[2] == 1
                                              return _idwt_inner_strided!(pl[1], ph[1], pw[1], 1, 1, 1, length(w), b, Val(:wavelets), Val(true))
                                          else
                                              return _idwt_inner_strided!(pl[1], ph[1], pw[1], pl[2], ph[2], pw[2], length(w), b, Val(:wavelets), Val(false))
                                          end
    end

    nw = length(w)
    m = length(l)
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    # scaling pass: window [2i-1, 2i+N-2], tail wraps.
    nplain = max(0, (nw - N + 2) >> 1)

    @inbounds for i in 1:min(m, nplain)
        j = 2 * i - 1
        li = l[i]
        if N <= 8
            @fastmath @views w[j:j+N-1] .+= li .* φ
        else
            @fastmath @simd for k in 1:N
                w[j + k - 1] += li * φ[k]
            end
        end
    end

    @inbounds for i in min(m, nplain)+1:m
        j = 2 * i - 1
        li = l[i]
        @fastmath for k in 1:N
            w[mod1(j + k - 1, nw)] += li * φ[k]
        end
    end

    # wavelet pass: window [2i-N+1, 2i], head wraps.
    nwrap = (N - 1) >> 1

    @inbounds for i in nwrap+1:m
        j = 2 * i - N + 1
        hi = h[i]
        if N <= 8
            @fastmath @views w[j:j+N-1] .+= hi .* ψ
        else
            @fastmath @simd for k in 1:N
                w[j + k - 1] += hi * ψ[k]
            end
        end
    end

    @inbounds for i in 1:min(m, nwrap)
        j = 2 * i - N + 1
        hi = h[i]
        @fastmath for k in 1:N
            w[mod1(j + k - 1, nw)] += hi * ψ[k]
        end
    end
    nothing
end

@fastfun function _idwt_inner_strided!(
    pl :: Ptr{T},
    ph :: Ptr{T},
    pw :: Ptr{T},
    sl :: Int,
    sh :: Int,
    sw :: Int,
    nw :: Int,
    b :: WTOrthogonalBasis{N, F},
    ::Val{:wavelets},
    ::Val{UNIT},
) :: Nothing where {T <: Number, N, F <: Number, UNIT}
    m = nw >> 1
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    # scaling pass: window [2i-1, 2i+N-2], tail wraps.
    nplain = max(0, (nw - N + 2) >> 1)

    if UNIT
        @inbounds for i in 1:min(m, nplain)
            j = 2 * i - 1
            li = unsafe_load(pl, i)
            if N <= 8
                @fastmath @simd for k in 1:N
                    off = j + k - 1
                    unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k], off)
                end
            else
                @fastmath for k in 1:N
                    off = j + k - 1
                    unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k], off)
                end
            end
        end

        @inbounds for i in min(m, nplain)+1:m
            j = 2 * i - 1
            li = unsafe_load(pl, i)
            @fastmath for k in 1:N
                off = mod1(j + k - 1, nw)
                unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k], off)
            end
        end

        # wavelet pass: window [2i-N+1, 2i], head wraps.
        nwrap = (N - 1) >> 1

        @inbounds for i in nwrap+1:m
            j = 2 * i - N + 1
            hi = unsafe_load(ph, i)
            if N <= 8
                @fastmath @simd for k in 1:N
                    off = j + k - 1
                    unsafe_store!(pw, unsafe_load(pw, off) + hi * ψ[k], off)
                end
            else
                @fastmath for k in 1:N
                    off = j + k - 1
                    unsafe_store!(pw, unsafe_load(pw, off) + hi * ψ[k], off)
                end
            end
        end

        @inbounds for i in 1:min(m, nwrap)
            j = 2 * i - N + 1
            hi = unsafe_load(ph, i)
            @fastmath for k in 1:N
                off = mod1(j + k - 1, nw)
                unsafe_store!(pw, unsafe_load(pw, off) + hi * ψ[k], off)
            end
        end
    else
        @inbounds for i in 1:min(m, nplain)
            j = 2 * i - 1
            li = unsafe_load(pl, (i - 1) * sl + 1)
            if N <= 8
                @fastmath @simd for k in 1:N
                    off = (j + k - 2) * sw + 1
                    unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k], off)
                end
            else
                @fastmath for k in 1:N
                    off = (j + k - 2) * sw + 1
                    unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k], off)
                end
            end
        end

        @inbounds for i in min(m, nplain)+1:m
            j = 2 * i - 1
            li = unsafe_load(pl, (i - 1) * sl + 1)
            @fastmath for k in 1:N
                off = (mod1(j + k - 1, nw) - 1) * sw + 1
                unsafe_store!(pw, unsafe_load(pw, off) + li * φ[k], off)
            end
        end

        # wavelet pass: window [2i-N+1, 2i], head wraps.
        nwrap = (N - 1) >> 1

        @inbounds for i in nwrap+1:m
            j = 2 * i - N + 1
            hi = unsafe_load(ph, (i - 1) * sh + 1)
            if N <= 8
                @fastmath @simd for k in 1:N
                    off = (j + k - 2) * sw + 1
                    unsafe_store!(pw, unsafe_load(pw, off) + hi * ψ[k], off)
                end
            else
                @fastmath for k in 1:N
                    off = (j + k - 2) * sw + 1
                    unsafe_store!(pw, unsafe_load(pw, off) + hi * ψ[k], off)
                end
            end
        end

        @inbounds for i in 1:min(m, nwrap)
            j = 2 * i - N + 1
            hi = unsafe_load(ph, (i - 1) * sh + 1)
            @fastmath for k in 1:N
                off = (mod1(j + k - 1, nw) - 1) * sw + 1
                unsafe_store!(pw, unsafe_load(pw, off) + hi * ψ[k], off)
            end
        end
    end
    nothing
end

