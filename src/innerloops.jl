# -----------------
# FORWARD TRANSFORM
# -----------------
#
# Two phase conventions are supported, dispatched on `Val{C}` (compile-time,
# zero runtime cost):
#
#   :aligned   -- the scaling and wavelet filters act on the same input
#                window: a[j] = Σ φ[k] x[2j-1+k-1],
#                        d[j] = Σ ψ[k] x[2j-1+k-1]   (mod 1 wrap-around).
#                This is the textbook phase (e.g. PyWavelets/Matlab-style)
#                and is the default.
#
#   :wavelets  -- the Wavelets.jl-compatible phase: the wavelet filter's
#                window starts N-2 taps before the scaling filter's window,
#                        d[j] = Σ ψ[k] x[2j+k-N]     (mod 1 wrap-around).
#                Equivalently the detail coefficients are the aligned ones
#                cyclically rotated by (N-2)/2 positions, so the wrap-around
#                sits at the *head* of the detail sequence instead of its
#                tail. For 2-tap filters the two conventions coincide.
#
# The scaling coefficients are identical in both conventions.
#
# The kernels below are "strided": they take the raw data pointer and the
# memory stride of each 1-D line, so that slices of higher-dimensional
# arrays (rows, columns, blocks) are transformed without per-element
# indexing overhead through the array wrapper. For arrays that do not
# expose a pointer (non-strided AbstractArrays) a generic fallback with
# ordinary indexing is used instead.

# Extract the (pointer, stride) of a 1-D array, or `nothing` if it is not
# strided. `Base.unsafe_convert` is defined for Array, SubArray and
# StridedView alike.
@inline _strided1(x::AbstractArray{T,1}) where {T} = begin
    s = strides(x)
    isempty(s) ? nothing : (Base.unsafe_convert(Ptr{T}, x), s[1])
end

# -----------------
# FORWARD TRANSFORM
# -----------------

# 2-tap transform: no mod1 branching.
@fastfun function _dwt_inner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{2, F},
    ::Val{:aligned},
) :: Nothing where {T <: Number, F <: Number}

    sx = _strided1(x)
    sw = _strided1(w)
    if sx !== nothing && sw !== nothing
        if sx[2] == 1 && sw[2] == 1
                                              return _dwt_inner_strided!(sx[1], sw[1], 1, 1, length(x), b, Val(:aligned), Val(true))
                                          else
                                              return _dwt_inner_strided!(sx[1], sw[1], sx[2], sw[2], length(x), b, Val(:aligned), Val(false))
                                          end
    end

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

# 2-tap: both conventions coincide (the ψ window equals the φ window).
@fastfun function _dwt_inner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{2, F},
    ::Val{:wavelets},
) :: Nothing where {T <: Number, F <: Number}
    _dwt_inner_loop!(x, w, b, Val(:aligned))
    nothing
end

@fastfun function _dwt_inner_strided!(
    px :: Ptr{T},
    pw :: Ptr{T},
    sx :: Int,
    sw :: Int,
    n :: Int,
    b :: WTOrthogonalBasis{2, F},
    ::Val{:aligned},
    ::Val{UNIT},
) :: Nothing where {T <: Number, F <: Number, UNIT}

    npairs = n >> 1
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    if UNIT
        @inbounds for (outi, sigi) in enumerate(1:2:n-1)
            x1 = unsafe_load(px, sigi)
            x2 = unsafe_load(px, sigi + 1)
            @fastmath unsafe_store!(pw, φ[1] * x1 + φ[2] * x2, outi)
            @fastmath unsafe_store!(pw, ψ[1] * x1 + ψ[2] * x2, npairs + outi)
        end
    else
        @inbounds for (outi, sigi) in enumerate(1:2:n-1)
            x1 = unsafe_load(px, (sigi - 1) * sx + 1)
            x2 = unsafe_load(px, sigi * sx + 1)
            @fastmath unsafe_store!(pw, φ[1] * x1 + φ[2] * x2, (outi - 1) * sw + 1)
            @fastmath unsafe_store!(pw, ψ[1] * x1 + ψ[2] * x2, (npairs + outi - 1) * sw + 1)
        end
    end
    nothing
end

# N-tap transform: mod1 branching only in the wrap-around tail.
@fastfun function _dwt_inner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{N, F},
    ::Val{:aligned},
) :: Nothing where {T <: Number, F <: Number, N}

    sx = _strided1(x)
    sw = _strided1(w)
    if sx !== nothing && sw !== nothing
        if sx[2] == 1 && sw[2] == 1
                                              return _dwt_inner_strided!(sx[1], sw[1], 1, 1, length(x), b, Val(:aligned), Val(true))
                                          else
                                              return _dwt_inner_strided!(sx[1], sw[1], sx[2], sw[2], length(x), b, Val(:aligned), Val(false))
                                          end
    end

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

# N-tap strided fast path. `px`/`pw` point at the first element of the
# input/output lines and `sx`/`sw` are their memory strides.
@fastfun function _dwt_inner_strided!(
    px :: Ptr{T},
    pw :: Ptr{T},
    sx :: Int,
    sw :: Int,
    n :: Int,
    b :: WTOrthogonalBasis{N, F},
    ::Val{:aligned},
    ::Val{UNIT},
) :: Nothing where {T <: Number, F <: Number, N, UNIT}

    npairs = n >> 1
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    nplain = max(0, (n - N + 2) >> 1)

    if UNIT
        @inbounds for outi in 1:nplain
            sigi = 2 * outi - 1
            sφ = zero(T)
            sψ = zero(T)
            if N <= 8
                @fastmath @simd for k in 1:N
                    xv = unsafe_load(px, sigi + k - 1)
                    sφ += φ[k] * xv
                    sψ += ψ[k] * xv
                end
            else
                @fastmath for k in 1:N
                    xv = unsafe_load(px, sigi + k - 1)
                    sφ += φ[k] * xv
                    sψ += ψ[k] * xv
                end
            end
            unsafe_store!(pw, sφ, outi)
            unsafe_store!(pw, sψ, npairs + outi)
        end

        @inbounds for outi in nplain+1:npairs
            sigi = 2 * outi - 1
            sφ = zero(T)
            sψ = zero(T)
            @fastmath for k in 1:N
                xv = unsafe_load(px, mod1(sigi + k - 1, n))
                sφ += φ[k] * xv
                sψ += ψ[k] * xv
            end
            unsafe_store!(pw, sφ, outi)
            unsafe_store!(pw, sψ, npairs + outi)
        end
    else
        @inbounds for outi in 1:nplain
            sigi = 2 * outi - 1
            sφ = zero(T)
            sψ = zero(T)
            if N <= 8
                @fastmath @simd for k in 1:N
                    xv = unsafe_load(px, (sigi + k - 2) * sx + 1)
                    sφ += φ[k] * xv
                    sψ += ψ[k] * xv
                end
            else
                @fastmath for k in 1:N
                    xv = unsafe_load(px, (sigi + k - 2) * sx + 1)
                    sφ += φ[k] * xv
                    sψ += ψ[k] * xv
                end
            end
            unsafe_store!(pw, sφ, (outi - 1) * sw + 1)
            unsafe_store!(pw, sψ, (npairs + outi - 1) * sw + 1)
        end

        @inbounds for outi in nplain+1:npairs
            sigi = 2 * outi - 1
            sφ = zero(T)
            sψ = zero(T)
            @fastmath for k in 1:N
                xv = unsafe_load(px, (mod1(sigi + k - 1, n) - 1) * sx + 1)
                sφ += φ[k] * xv
                sψ += ψ[k] * xv
            end
            unsafe_store!(pw, sφ, (outi - 1) * sw + 1)
            unsafe_store!(pw, sψ, (npairs + outi - 1) * sw + 1)
        end
    end
    nothing
end

# N-tap transform, Wavelets.jl-compatible phase: the wavelet filter's
# window starts N-2 taps before the scaling filter's window, so the
# wrap-around sits at the head of the detail sequence (first nwrap outputs)
# instead of its tail. The scaling coefficients are unchanged.
@fastfun function _dwt_inner_loop!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis{N, F},
    ::Val{:wavelets},
) :: Nothing where {T <: Number, F <: Number, N}

    sx = _strided1(x)
    sw = _strided1(w)
    if sx !== nothing && sw !== nothing
        if sx[2] == 1 && sw[2] == 1
                                              return _dwt_inner_strided!(sx[1], sw[1], 1, 1, length(x), b, Val(:wavelets), Val(true))
                                          else
                                              return _dwt_inner_strided!(sx[1], sw[1], sx[2], sw[2], length(x), b, Val(:wavelets), Val(false))
                                          end
    end

    ls = @view w[1:end>>1]
    hs = @view w[(end>>1)+1:end]
    n = length(x)
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    # scaling coefficients: aligned window [2outi-1, 2outi+N-2], tail wraps.
    nplain = max(0, (n - N + 2) >> 1)

    @inbounds for outi in 1:nplain
        sigi = 2 * outi - 1
        sφ = zero(T)
        if N <= 8
            @fastmath @simd for k in 1:N
                sφ += φ[k] * x[sigi + k - 1]
            end
        else
            @fastmath for k in 1:N
                sφ += φ[k] * x[sigi + k - 1]
            end
        end
        ls[outi] = sφ
    end

    @inbounds for outi in nplain+1:n>>1
        sigi = 2 * outi - 1
        sφ = zero(T)
        @fastmath for k in 1:N
            sφ += φ[k] * x[mod1(sigi + k - 1, n)]
        end
        ls[outi] = sφ
    end

    # wavelet coefficients: window [2outi-N+1, 2outi], head wraps.
    nwrap = (N - 1) >> 1

    @inbounds for outi in nwrap+1:n>>1
        sigi = 2 * outi - N + 1
        sψ = zero(T)
        if N <= 8
            @fastmath @simd for k in 1:N
                sψ += ψ[k] * x[sigi + k - 1]
            end
        else
            @fastmath for k in 1:N
                sψ += ψ[k] * x[sigi + k - 1]
            end
        end
        hs[outi] = sψ
    end

    @inbounds for outi in 1:min(nwrap, n>>1)
        sigi = 2 * outi - N + 1
        sψ = zero(T)
        @fastmath for k in 1:N
            sψ += ψ[k] * x[mod1(sigi + k - 1, n)]
        end
        hs[outi] = sψ
    end
    nothing
end

@fastfun function _dwt_inner_strided!(
    px :: Ptr{T},
    pw :: Ptr{T},
    sx :: Int,
    sw :: Int,
    n :: Int,
    b :: WTOrthogonalBasis{N, F},
    ::Val{:wavelets},
    ::Val{UNIT},
) :: Nothing where {T <: Number, F <: Number, N, UNIT}

    npairs = n >> 1
    φ = T.(b.φ)
    ψ = T.(b.ψ)

    # scaling coefficients: aligned window [2outi-1, 2outi+N-2], tail wraps.
    nplain = max(0, (n - N + 2) >> 1)

    if UNIT
        @inbounds for outi in 1:nplain
            sigi = 2 * outi - 1
            sφ = zero(T)
            if N <= 8
                @fastmath @simd for k in 1:N
                    sφ += φ[k] * unsafe_load(px, sigi + k - 1)
                end
            else
                @fastmath for k in 1:N
                    sφ += φ[k] * unsafe_load(px, sigi + k - 1)
                end
            end
            unsafe_store!(pw, sφ, outi)
        end

        @inbounds for outi in nplain+1:npairs
            sigi = 2 * outi - 1
            sφ = zero(T)
            @fastmath for k in 1:N
                sφ += φ[k] * unsafe_load(px, mod1(sigi + k - 1, n))
            end
            unsafe_store!(pw, sφ, outi)
        end

        # wavelet coefficients: window [2outi-N+1, 2outi], head wraps.
        nwrap = (N - 1) >> 1

        @inbounds for outi in nwrap+1:npairs
            sigi = 2 * outi - N + 1
            sψ = zero(T)
            if N <= 8
                @fastmath @simd for k in 1:N
                    sψ += ψ[k] * unsafe_load(px, sigi + k - 1)
                end
            else
                @fastmath for k in 1:N
                    sψ += ψ[k] * unsafe_load(px, sigi + k - 1)
                end
            end
            unsafe_store!(pw, sψ, npairs + outi)
        end

        @inbounds for outi in 1:min(nwrap, npairs)
            sigi = 2 * outi - N + 1
            sψ = zero(T)
            @fastmath for k in 1:N
                sψ += ψ[k] * unsafe_load(px, mod1(sigi + k - 1, n))
            end
            unsafe_store!(pw, sψ, npairs + outi)
        end
    else
        @inbounds for outi in 1:nplain
            sigi = 2 * outi - 1
            sφ = zero(T)
            if N <= 8
                @fastmath @simd for k in 1:N
                    sφ += φ[k] * unsafe_load(px, (sigi + k - 2) * sx + 1)
                end
            else
                @fastmath for k in 1:N
                    sφ += φ[k] * unsafe_load(px, (sigi + k - 2) * sx + 1)
                end
            end
            unsafe_store!(pw, sφ, (outi - 1) * sw + 1)
        end

        @inbounds for outi in nplain+1:npairs
            sigi = 2 * outi - 1
            sφ = zero(T)
            @fastmath for k in 1:N
                sφ += φ[k] * unsafe_load(px, (mod1(sigi + k - 1, n) - 1) * sx + 1)
            end
            unsafe_store!(pw, sφ, (outi - 1) * sw + 1)
        end

        # wavelet coefficients: window [2outi-N+1, 2outi], head wraps.
        nwrap = (N - 1) >> 1

        @inbounds for outi in nwrap+1:npairs
            sigi = 2 * outi - N + 1
            sψ = zero(T)
            if N <= 8
                @fastmath @simd for k in 1:N
                    sψ += ψ[k] * unsafe_load(px, (sigi + k - 2) * sx + 1)
                end
            else
                @fastmath for k in 1:N
                    sψ += ψ[k] * unsafe_load(px, (sigi + k - 2) * sx + 1)
                end
            end
            unsafe_store!(pw, sψ, (npairs + outi - 1) * sw + 1)
        end

        @inbounds for outi in 1:min(nwrap, npairs)
            sigi = 2 * outi - N + 1
            sψ = zero(T)
            @fastmath for k in 1:N
                sψ += ψ[k] * unsafe_load(px, (mod1(sigi + k - 1, n) - 1) * sx + 1)
            end
            unsafe_store!(pw, sψ, (npairs + outi - 1) * sw + 1)
        end
    end
    nothing
end

# Discrete wavelet transform, 1-D
@fastfun function _dwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    level=1,
    ::Val{C}=Val(:aligned);
    wpt=false
) :: AbstractArray{T,1} where {T <: Number, C}

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
                Val{C}(),
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
    level=1,
    ::Val{C}=Val(:aligned);
    wpt=false
) :: AbstractArray{T,1} where {T <: Number, C}

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
            _idwt_inner_loop!(lv, hv, dv, b, Val{C}())
        end
        src, dst = dst, src
        nreal += 1
    end

    isodd(nreal) && copyto!(x, w)

    return x
end
