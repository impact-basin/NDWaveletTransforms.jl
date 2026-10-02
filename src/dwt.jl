@generated function zipslices_1d(x::T, w::T) where {E, N, T <: AbstractArray{E,N}}
    quote zip(
        eachslice(x, dims = $N),
        eachslice(w, dims = $N),
    ) end
end

@generated function zipslices_nd(x::T, w::T) where {E, N, T <: AbstractArray{E,N}}
    dims = ntuple(i -> i, N-1)
    quote zip(
        eachslice(x, dims = $dims),
        eachslice(w, dims = $dims),
    ) end
end

macro maybe_thread(s::Symbol, expr...)
    return quote
        if $s
            @inbounds @floop $(expr...)
        else
            @inbounds $(expr...)
        end
    end |> esc
end

# Threading only pays off once a pass has enough work: a threaded pass pays
# a fixed task-spawn/barrier cost (tens of us), so small passes run faster
# single-threaded. This is the minimum number of filter taps applied
# (lines x pairs x ntaps) before a pass is threaded.
const _THREAD_MIN_WORK = 500_000

@fastfun function dwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt = false,
    top = false,
    convention = :aligned
) :: A where {T <: Number, A <: AbstractArray{T,1}}

    is_gpu(x) ? _dwt_gpu!(x, w, b, l, Val(convention), wpt = wpt) :
                _dwt!(x, w, b, l[1], Val(convention), wpt = wpt)
    return x
end


@fastfun function dwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt = false,
    top = true,
    convention = :aligned
) :: A where {T <: Number, N, A <: AbstractArray{T,N}}

    is_gpu(x) && return _dwt_gpu!(x, w, b, l, Val(convention), wpt = wpt)
    return _dwt_nd!(x, w, b, l, Val(convention); wpt = wpt, top = top)
end

@fastfun function _dwt_nd!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector,
    ::Val{C};
    wpt = false,
    top = true
) :: A where {C, T <: Number, N, A <: AbstractArray{T,N}}

    dothread = top && Threads.nthreads() > 1 && length(x) * length(b.φ) >= _THREAD_MIN_WORK

    # Level-1 passes along every axis with l[a] >= 1, in axis order. The
    # fused passes operate on the whole N-D region (no per-line slicing),
    # ping-ponging between x and w (write-through, no copyback).
    datanow = 1
    for a in 1:N
        (l[a] >= 1 && size(x, a) >= 2) || continue
        if datanow == 1
            _dwt_axis_pass!(x, w, b, a, Val{C}(); top = top)
            datanow = 2
        else
            _dwt_axis_pass!(w, x, b, a, Val{C}(); top = top)
            datanow = 1
        end
    end

    lm1 = l .- 1
    if any(lm1 .> 0)
        if datanow == 1
            @maybe_thread dothread for subspace in subspaces(w, x, wpt)
                wss, xss = subspace
                _dwt_nd!(xss, wss, b, lm1, Val{C}(); wpt = wpt, top=false)
            end
        else
            @maybe_thread dothread for subspace in subspaces(x, w, wpt)
                wss, xss = subspace
                _dwt_nd!(xss, wss, b, lm1, Val{C}(); wpt = wpt, top=false)
            end
        end
    end

    datanow == 2 && copyto!(x, w)
    return x
end

@fastfun function idwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt=false,
    top=false,
    convention = :aligned
) :: A where {T <: Number, A <: AbstractArray{T,1}}

    is_gpu(x) && return _idwt_gpu!(x, w, b, l, Val(convention), wpt = wpt)
    return _idwt!(x, w, b, l[1], Val(convention), wpt=wpt)
end

@fastfun function idwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt=false,
    top=true,
    convention = :aligned
) :: A where {T <: Number, N, A <: AbstractArray{T,N}}

    is_gpu(x) && return _idwt_gpu!(x, w, b, l, Val(convention), wpt = wpt)
    return _idwt_nd!(x, w, b, l, Val(convention); wpt = wpt, top = top)
end

@fastfun function _idwt_nd!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector,
    ::Val{C};
    wpt=false,
    top=true
) :: A where {C, T <: Number, N, A <: AbstractArray{T,N}}

    dothread = top && Threads.nthreads() > 1 && length(x) * length(b.φ) >= _THREAD_MIN_WORK

    lm1 = l .- 1
    any(lm1 .> 0) && @maybe_thread dothread for subspace in subspaces(w, x, wpt)
        wss, xss = subspace
        _idwt_nd!(xss, wss, b, lm1, Val{C}(); wpt = wpt, top=false)
    end

    # Level-1 passes along every axis with l[a] >= 1, in reverse axis
    # order (the inverse of the forward analysis).
    datanow = 1
    for a in N:-1:1
        (l[a] >= 1 && size(x, a) >= 2) || continue
        if datanow == 1
            _idwt_axis_pass!(x, w, b, a, Val{C}(); top = top)
            datanow = 2
        else
            _idwt_axis_pass!(w, x, b, a, Val{C}(); top = top)
            datanow = 1
        end
    end

    datanow == 2 && copyto!(x, w)
    return x
end

# ---------------------------------------------------------------------------
# Fused one-level passes along one axis of an N-D region, used by the N-D
# drivers. Instead of slicing every line (which forces per-element indexing
# through a strided array wrapper and one call per line), the pass reads and
# writes the region directly through raw pointers. The inner loop always
# runs along the memory-contiguous direction:
#   * axis 1 (unit line stride): line-outer, pair-inner;
#   * other axes: pair-outer, line-inner, with the innermost loop along
#     dim 1, which is contiguous across lines of a column-major region.
# `Val{CONTIG}` is chosen once per pass from the strides, so both hot loops
# are fully specialised and vectorisable.
# ---------------------------------------------------------------------------

@fastfun function _dwt_axis_pass!(x, w, b, ax, ::Val{C}; top = true) where {C}
    _dwt_axis_pass!(x, w, b, ax, Val{C}(), Val(strides(x)[ax] == 1); top = top)
    nothing
end

# Linear offset of line `c` (a CartesianIndex over the dims kept in `outer`,
# with the dropped dims having extent 1) within the region.
@inline function _line_base(::Val{N}, s, c, ax) where {N}
    base = 1
    for d in 1:N
        (d == 1 || d == ax) && continue
        base += (c[d] - 1) * s[d]
    end
    base
end

# Number of other-axis coordinates processed per cache block in the strided
# passes. Each coordinate's data spans n1 x n x sizeof(T) bytes; the block
# footprint times the active threads must stay within the shared cache so
# the overlapping tap re-reads hit cache instead of DRAM.
const _PASS_BLOCK = 8

# 1-based linear offset of the line with flattened other-coordinate index f
# (dims other than 1 and ax, dim 2 fastest).
@inline function _block_base(s, odims, odsize, f)
    rem = f - 1
    base = 1
    for j in eachindex(odims)
        c = mod(rem, odsize[j]) + 1
        base += (c - 1) * s[odims[j]]
        rem = rem ÷ odsize[j]
    end
    base
end

@fastfun function _dwt_axis_pass!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    ax :: Int,
    ::Val{:aligned},
    ::Val{CONTIG};
    top = true
) :: Nothing where {T <: Number, N, CONTIG}

    n = size(x, ax)
    npairs = n >> 1
    s = strides(x)
    sa = s[ax]
    px = Base.unsafe_convert(Ptr{T}, x)
    pw = Base.unsafe_convert(Ptr{T}, w)
    φ = T.(b.φ)
    ψ = T.(b.ψ)
    ntaps = length(φ)
    nplain = max(0, (n - ntaps + 2) >> 1)
    dothread = top && Threads.nthreads() > 1 && length(x) * ntaps >= _THREAD_MIN_WORK

    if CONTIG
        # lines (along axis 1) are memory-contiguous: line-outer, pair-inner.
        @maybe_thread dothread for c in CartesianIndices(ntuple(d -> d == 1 ? 1 : size(x, d), N))
            base = _line_base(Val(N), s, c, 1)
            @simd for p in 1:nplain
                sφ = zero(T)
                sψ = zero(T)
                @fastmath for k in 1:ntaps
                    xv = unsafe_load(px, base + 2p + k - 3)
                    sφ += φ[k] * xv
                    sψ += ψ[k] * xv
                end
                unsafe_store!(pw, sφ, base + p - 1)
                unsafe_store!(pw, sψ, base + npairs + p - 1)
            end
            @simd for p in nplain+1:npairs
                sφ = zero(T)
                sψ = zero(T)
                @fastmath for k in 1:ntaps
                    xv = unsafe_load(px, base + mod1(2p - 1 + k - 1, n) - 1)
                    sφ += φ[k] * xv
                    sψ += ψ[k] * xv
                end
                unsafe_store!(pw, sφ, base + p - 1)
                unsafe_store!(pw, sψ, base + npairs + p - 1)
            end
        end
    else
        # Strided lines. The overlapping tap windows re-read each element
        # ~ntaps/2 times, so the pair loop is blocked over the other-axis
        # coordinates: each block's data stays cache-resident across the
        # pair loop and the re-reads come from cache instead of DRAM.
        n1 = size(x, 1)
        odims = Tuple(d for d in 1:N if d != 1 && d != ax)
        odsize = Tuple(size(x, d) for d in odims)
        ngrid = prod(odsize)
        nblocks = cld(ngrid, _PASS_BLOCK)
        @maybe_thread dothread for blk in 1:nblocks
            f0 = (blk - 1) * _PASS_BLOCK + 1
            f1 = min(blk * _PASS_BLOCK, ngrid)
            for p in 1:npairs
                if p <= nplain
                    for f in f0:f1
                        base = _block_base(s, odims, odsize, f)
                        @simd for i1 in 1:n1
                            sφ = zero(T)
                            sψ = zero(T)
                            @fastmath for k in 1:ntaps
                                xv = unsafe_load(px, base + i1 - 1 + (2p - 2 + k - 1) * sa)
                                sφ += φ[k] * xv
                                sψ += ψ[k] * xv
                            end
                            unsafe_store!(pw, sφ, base + i1 - 1 + (p - 1) * sa)
                            unsafe_store!(pw, sψ, base + i1 - 1 + (npairs + p - 1) * sa)
                        end
                    end
                else
                    for f in f0:f1
                        base = _block_base(s, odims, odsize, f)
                        @simd for i1 in 1:n1
                            sφ = zero(T)
                            sψ = zero(T)
                            @fastmath for k in 1:ntaps
                                xv = unsafe_load(px, base + i1 - 1 + (mod1(2p - 1 + k - 1, n) - 1) * sa)
                                sφ += φ[k] * xv
                                sψ += ψ[k] * xv
                            end
                            unsafe_store!(pw, sφ, base + i1 - 1 + (p - 1) * sa)
                            unsafe_store!(pw, sψ, base + i1 - 1 + (npairs + p - 1) * sa)
                        end
                    end
                end
            end
        end
    end
    nothing
end

# Wavelets.jl-compatible phase: the scaling coefficients use the aligned
# window (tail wrap) while the wavelet coefficients use the shifted window
# (head wrap), so the φ and ψ contributions are emitted in separate loops.
@fastfun function _dwt_axis_pass!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    ax :: Int,
    ::Val{:wavelets},
    ::Val{CONTIG};
    top = true
) :: Nothing where {T <: Number, N, CONTIG}

    n = size(x, ax)
    npairs = n >> 1
    s = strides(x)
    sa = s[ax]
    px = Base.unsafe_convert(Ptr{T}, x)
    pw = Base.unsafe_convert(Ptr{T}, w)
    φ = T.(b.φ)
    ψ = T.(b.ψ)
    ntaps = length(φ)
    nplain = max(0, (n - ntaps + 2) >> 1)
    nwrap = (ntaps - 1) >> 1
    dothread = top && Threads.nthreads() > 1 && length(x) * ntaps >= _THREAD_MIN_WORK

    if CONTIG
        @maybe_thread dothread for c in CartesianIndices(ntuple(d -> d == 1 ? 1 : size(x, d), N))
            base = _line_base(Val(N), s, c, 1)
            @simd for p in 1:nplain
                sφ = zero(T)
                @fastmath for k in 1:ntaps
                    sφ += φ[k] * unsafe_load(px, base + 2p + k - 3)
                end
                unsafe_store!(pw, sφ, base + p - 1)
            end
            @simd for p in nplain+1:npairs
                sφ = zero(T)
                @fastmath for k in 1:ntaps
                    sφ += φ[k] * unsafe_load(px, base + mod1(2p - 1 + k - 1, n) - 1)
                end
                unsafe_store!(pw, sφ, base + p - 1)
            end
            @simd for p in 1:min(nwrap, npairs)
                sψ = zero(T)
                @fastmath for k in 1:ntaps
                    sψ += ψ[k] * unsafe_load(px, base + mod1(2p - ntaps + k, n) - 1)
                end
                unsafe_store!(pw, sψ, base + npairs + p - 1)
            end
            @simd for p in max(1, nwrap+1):npairs
                sψ = zero(T)
                @fastmath for k in 1:ntaps
                    sψ += ψ[k] * unsafe_load(px, base + 2p - ntaps + k - 1)
                end
                unsafe_store!(pw, sψ, base + npairs + p - 1)
            end
        end
    else
        n1 = size(x, 1)
        odims = Tuple(d for d in 1:N if d != 1 && d != ax)
        odsize = Tuple(size(x, d) for d in odims)
        ngrid = prod(odsize)
        nblocks = cld(ngrid, _PASS_BLOCK)
        @maybe_thread dothread for blk in 1:nblocks
            f0 = (blk - 1) * _PASS_BLOCK + 1
            f1 = min(blk * _PASS_BLOCK, ngrid)
            for p in 1:npairs
                if p <= nplain
                    for f in f0:f1
                        base = _block_base(s, odims, odsize, f)
                        @simd for i1 in 1:n1
                            sφ = zero(T)
                            @fastmath for k in 1:ntaps
                                sφ += φ[k] * unsafe_load(px, base + i1 - 1 + (2p - 2 + k - 1) * sa)
                            end
                            unsafe_store!(pw, sφ, base + i1 - 1 + (p - 1) * sa)
                        end
                    end
                else
                    for f in f0:f1
                        base = _block_base(s, odims, odsize, f)
                        @simd for i1 in 1:n1
                            sφ = zero(T)
                            @fastmath for k in 1:ntaps
                                sφ += φ[k] * unsafe_load(px, base + i1 - 1 + (mod1(2p - 1 + k - 1, n) - 1) * sa)
                            end
                            unsafe_store!(pw, sφ, base + i1 - 1 + (p - 1) * sa)
                        end
                    end
                end
                if p <= nwrap
                    for f in f0:f1
                        base = _block_base(s, odims, odsize, f)
                        @simd for i1 in 1:n1
                            sψ = zero(T)
                            @fastmath for k in 1:ntaps
                                sψ += ψ[k] * unsafe_load(px, base + i1 - 1 + (mod1(2p - ntaps + k, n) - 1) * sa)
                            end
                            unsafe_store!(pw, sψ, base + i1 - 1 + (npairs + p - 1) * sa)
                        end
                    end
                else
                    for f in f0:f1
                        base = _block_base(s, odims, odsize, f)
                        @simd for i1 in 1:n1
                            sψ = zero(T)
                            @fastmath for k in 1:ntaps
                                sψ += ψ[k] * unsafe_load(px, base + i1 - 1 + (2p - ntaps + k - 1) * sa)
                            end
                            unsafe_store!(pw, sψ, base + i1 - 1 + (npairs + p - 1) * sa)
                        end
                    end
                end
            end
        end
    end
    nothing
end

# Inverse one-level pass along one axis of an N-D region. The synthesis is
# transposed relative to the forward analysis: for each tap k the
# coefficient pairs feeding positions 2i+k-2 (mod n) are disjoint, so the
# k-outer scatter loops are dependency-free and vectorisable. The output
# region is zeroed once before the accumulation. For threaded strided
# passes, positions are gathered instead (write-once, disjoint over j).
@fastfun function _idwt_axis_pass!(x, w, b, ax, ::Val{C}; top = true) where {C}
    _idwt_axis_pass!(x, w, b, ax, Val{C}(), Val(strides(x)[ax] == 1); top = top)
    nothing
end

@fastfun function _idwt_axis_pass!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    ax :: Int,
    ::Val{:aligned},
    ::Val{CONTIG};
    top = true
) :: Nothing where {T <: Number, N, CONTIG}

    n = size(x, ax)
    m = n >> 1
    s = strides(x)
    sa = s[ax]
    px = Base.unsafe_convert(Ptr{T}, x)
    pw = Base.unsafe_convert(Ptr{T}, w)
    φ = T.(b.φ)
    ψ = T.(b.ψ)
    ntaps = length(φ)
    dothread = top && Threads.nthreads() > 1 && length(x) * ntaps >= _THREAD_MIN_WORK

    if CONTIG
        @maybe_thread dothread for c in CartesianIndices(ntuple(d -> d == 1 ? 1 : size(x, d), N))
            base = _line_base(Val(N), s, c, 1)
            @simd for j in 0:n-1
                unsafe_store!(pw, zero(T), base + j)
            end
            for k in 1:ntaps
                φk = φ[k]
                ψk = ψ[k]
                nplain = max(0, min(fld(n - k + 2, 2), m))
                @simd for i in 1:nplain
                    off = base + 2i + k - 3
                    unsafe_store!(pw, unsafe_load(pw, off) +
                                   unsafe_load(px, base + i - 1) * φk +
                                   unsafe_load(px, base + m + i - 1) * ψk, off)
                end
                @simd for i in max(1, nplain+1):m
                    off = base + mod1(2i + k - 2, n) - 1
                    unsafe_store!(pw, unsafe_load(pw, off) +
                                   unsafe_load(px, base + i - 1) * φk +
                                   unsafe_load(px, base + m + i - 1) * ψk, off)
                end
            end
        end
    elseif dothread
        # strided lines, threaded: gather over output positions. Each
        # position is written exactly once (no zeroing needed) and
        # positions are disjoint, so threading over j is race-free. The
        # other-axis grid is blocked so the coefficient re-reads across j
        # hit cache.
        n1 = size(x, 1)
        odims = Tuple(d for d in 1:N if d != 1 && d != ax)
        odsize = Tuple(size(x, d) for d in odims)
        ngrid = prod(odsize)
        nblocks = cld(ngrid, _PASS_BLOCK)
        tmax_t = (ntaps + n - 1) ÷ n
        φv = Vector{T}(φ)
        ψv = Vector{T}(ψ)
        @maybe_thread dothread for blk in 1:nblocks
            f0 = (blk - 1) * _PASS_BLOCK + 1
            f1 = min(blk * _PASS_BLOCK, ngrid)
            for j in 1:n
                for f in f0:f1
                    base = _block_base(s, odims, odsize, f)
                    @simd for i1 in 1:n1
                        acc = zero(T)
                        for t in 0:tmax_t
                            ilo = max(1, cld(j + t * n - ntaps + 2, 2))
                            ihi = min(m, fld(j + t * n + 1, 2))
                            @fastmath for i in ilo:ihi
                                k = j + t * n - 2i + 2
                                acc += unsafe_load(px, base + i1 - 1 + (i - 1) * sa) * φv[k] +
                                       unsafe_load(px, base + i1 - 1 + (m + i - 1) * sa) * ψv[k]
                            end
                        end
                        unsafe_store!(pw, acc, base + i1 - 1 + (j - 1) * sa)
                    end
                end
            end
        end
    else
        # strided lines, single-threaded: the transposed k-outer scatter
        # vectorises like the forward pass, blocked over the other-axis
        # grid so the read-modify-write re-reads hit cache.
        n1 = size(x, 1)
        odims = Tuple(d for d in 1:N if d != 1 && d != ax)
        odsize = Tuple(size(x, d) for d in odims)
        ngrid = prod(odsize)
        nblocks = cld(ngrid, _PASS_BLOCK)
        for blk in 1:nblocks
            f0 = (blk - 1) * _PASS_BLOCK + 1
            f1 = min(blk * _PASS_BLOCK, ngrid)
            for f in f0:f1
                base = _block_base(s, odims, odsize, f)
                # zero every line: all n1 dim-1 positions of each of the n
                # line positions
                for j in 0:n-1
                    @simd for i1 in 1:n1
                        unsafe_store!(pw, zero(T), base + i1 - 1 + j * sa)
                    end
                end
            end
            for k in 1:ntaps
                φk = φ[k]
                ψk = ψ[k]
                nplain = max(0, min(fld(n - k + 2, 2), m))
                for f in f0:f1
                    base = _block_base(s, odims, odsize, f)
                    @inbounds for i in 1:nplain
                        @simd for i1 in 1:n1
                            off = base + i1 - 1 + (2i - 2 + k - 1) * sa
                            unsafe_store!(pw, unsafe_load(pw, off) +
                                           unsafe_load(px, base + i1 - 1 + (i - 1) * sa) * φk +
                                           unsafe_load(px, base + i1 - 1 + (m + i - 1) * sa) * ψk, off)
                        end
                    end
                    @inbounds for i in max(1, nplain+1):m
                        @simd for i1 in 1:n1
                            off = base + i1 - 1 + (mod1(2i - 1 + k - 1, n) - 1) * sa
                            unsafe_store!(pw, unsafe_load(pw, off) +
                                           unsafe_load(px, base + i1 - 1 + (i - 1) * sa) * φk +
                                           unsafe_load(px, base + i1 - 1 + (m + i - 1) * sa) * ψk, off)
                        end
                    end
                end
            end
        end
    end
    nothing
end

# Wavelets.jl-compatible phase: the scaling coefficients feed the aligned
# window [2i-1, 2i+N-2] (tail wraps), the wavelet coefficients the shifted
# window [2i-N+1, 2i] (head wraps).
@fastfun function _idwt_axis_pass!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    ax :: Int,
    ::Val{:wavelets},
    ::Val{CONTIG};
    top = true
) :: Nothing where {T <: Number, N, CONTIG}

    n = size(x, ax)
    m = n >> 1
    s = strides(x)
    sa = s[ax]
    px = Base.unsafe_convert(Ptr{T}, x)
    pw = Base.unsafe_convert(Ptr{T}, w)
    φ = T.(b.φ)
    ψ = T.(b.ψ)
    ntaps = length(φ)
    dothread = top && Threads.nthreads() > 1 && length(x) * ntaps >= _THREAD_MIN_WORK

    if CONTIG
        @maybe_thread dothread for c in CartesianIndices(ntuple(d -> d == 1 ? 1 : size(x, d), N))
            base = _line_base(Val(N), s, c, 1)
            @simd for j in 0:n-1
                unsafe_store!(pw, zero(T), base + j)
            end
            # scaling pass (aligned window, tail wraps)
            for k in 1:ntaps
                φk = φ[k]
                nplain = max(0, min(fld(n - k + 2, 2), m))
                @simd for i in 1:nplain
                    off = base + 2i + k - 3
                    unsafe_store!(pw, unsafe_load(pw, off) +
                                   unsafe_load(px, base + i - 1) * φk, off)
                end
                @simd for i in max(1, nplain+1):m
                    off = base + mod1(2i + k - 2, n) - 1
                    unsafe_store!(pw, unsafe_load(pw, off) +
                                   unsafe_load(px, base + i - 1) * φk, off)
                end
            end
            # wavelet pass (shifted window [2i-N+1, 2i], head wraps)
            for k in 1:ntaps
                ψk = ψ[k]
                nwrapk = min(fld(ntaps - k, 2), m)
                @simd for i in 1:nwrapk
                    off = base + mod1(2i - ntaps + k, n) - 1
                    unsafe_store!(pw, unsafe_load(pw, off) +
                                   unsafe_load(px, base + m + i - 1) * ψk, off)
                end
                @simd for i in nwrapk+1:m
                    off = base + 2i - ntaps + k - 1
                    unsafe_store!(pw, unsafe_load(pw, off) +
                                   unsafe_load(px, base + m + i - 1) * ψk, off)
                end
            end
        end
    elseif dothread
        # strided lines, threaded: gather over output positions
        # (write-once, race-free over j), blocked over the other-axis grid.
        n1 = size(x, 1)
        odims = Tuple(d for d in 1:N if d != 1 && d != ax)
        odsize = Tuple(size(x, d) for d in odims)
        ngrid = prod(odsize)
        nblocks = cld(ngrid, _PASS_BLOCK)
        tmax_t = (ntaps + n - 1) ÷ n
        φv = Vector{T}(φ)
        ψv = Vector{T}(ψ)
        @maybe_thread dothread for blk in 1:nblocks
            f0 = (blk - 1) * _PASS_BLOCK + 1
            f1 = min(blk * _PASS_BLOCK, ngrid)
            for j in 1:n
                for f in f0:f1
                    base = _block_base(s, odims, odsize, f)
                    @simd for i1 in 1:n1
                        acc = zero(T)
                        # scaling pass
                        for t in 0:tmax_t
                            ilo = max(1, cld(j + t * n - ntaps + 2, 2))
                            ihi = min(m, fld(j + t * n + 1, 2))
                            @fastmath for i in ilo:ihi
                                k = j + t * n - 2i + 2
                                acc += unsafe_load(px, base + i1 - 1 + (i - 1) * sa) * φv[k]
                            end
                        end
                        # wavelet pass
                        for t in cld(3 - j - ntaps, n):fld(n + 1 - j, n)
                            ilo = max(1, cld(j + t * n, 2))
                            ihi = min(m, fld(j + t * n + ntaps - 1, 2))
                            @fastmath for i in ilo:ihi
                                k = j + t * n - 2i + ntaps
                                acc += unsafe_load(px, base + i1 - 1 + (m + i - 1) * sa) * ψv[k]
                            end
                        end
                        unsafe_store!(pw, acc, base + i1 - 1 + (j - 1) * sa)
                    end
                end
            end
        end
    else
        # strided lines, single-threaded: transposed k-outer scatter,
        # blocked over the other-axis grid.
        n1 = size(x, 1)
        odims = Tuple(d for d in 1:N if d != 1 && d != ax)
        odsize = Tuple(size(x, d) for d in odims)
        ngrid = prod(odsize)
        nblocks = cld(ngrid, _PASS_BLOCK)
        for blk in 1:nblocks
            f0 = (blk - 1) * _PASS_BLOCK + 1
            f1 = min(blk * _PASS_BLOCK, ngrid)
            for f in f0:f1
                base = _block_base(s, odims, odsize, f)
                # zero every line: all n1 dim-1 positions of each of the n
                # line positions
                for j in 0:n-1
                    @simd for i1 in 1:n1
                        unsafe_store!(pw, zero(T), base + i1 - 1 + j * sa)
                    end
                end
            end
            # scaling pass: aligned window, tail wraps
            for k in 1:ntaps
                φk = φ[k]
                nplain = max(0, min(fld(n - k + 2, 2), m))
                for f in f0:f1
                    base = _block_base(s, odims, odsize, f)
                    @inbounds for i in 1:nplain
                        @simd for i1 in 1:n1
                            off = base + i1 - 1 + (2i - 2 + k - 1) * sa
                            unsafe_store!(pw, unsafe_load(pw, off) +
                                           unsafe_load(px, base + i1 - 1 + (i - 1) * sa) * φk, off)
                        end
                    end
                    @inbounds for i in max(1, nplain+1):m
                        @simd for i1 in 1:n1
                            off = base + i1 - 1 + (mod1(2i - 1 + k - 1, n) - 1) * sa
                            unsafe_store!(pw, unsafe_load(pw, off) +
                                           unsafe_load(px, base + i1 - 1 + (i - 1) * sa) * φk, off)
                        end
                    end
                end
            end
            # wavelet pass: shifted window [2i-N+1, 2i], head wraps
            for k in 1:ntaps
                ψk = ψ[k]
                nwrapk = min(fld(ntaps - k, 2), m)
                for f in f0:f1
                    base = _block_base(s, odims, odsize, f)
                    @inbounds for i in 1:nwrapk
                        @simd for i1 in 1:n1
                            off = base + i1 - 1 + (mod1(2i - ntaps + k, n) - 1) * sa
                            unsafe_store!(pw, unsafe_load(pw, off) +
                                           unsafe_load(px, base + i1 - 1 + (m + i - 1) * sa) * ψk, off)
                        end
                    end
                    @inbounds for i in nwrapk+1:m
                        @simd for i1 in 1:n1
                            off = base + i1 - 1 + (2i - ntaps + k - 1) * sa
                            unsafe_store!(pw, unsafe_load(pw, off) +
                                           unsafe_load(px, base + i1 - 1 + (m + i - 1) * sa) * ψk, off)
                        end
                    end
                end
            end
        end
    end
    nothing
end

@fastfun function dwt!(x::A, b, l :: Int; wpt = false, convention = :aligned) :: A where {T, N, A <: AbstractArray{T,N}}
    w = similar(x)
    if is_gpu(x)
        dwt!(x, w, b, repeat([l], N); wpt = wpt, convention = convention)
    else
        dwt!(StridedView(x), StridedView(w), b, repeat([l], N); wpt = wpt, convention = convention)
    end
end

@fastfun function dwt!(x::T, b, l; wpt = false, convention = :aligned) :: T where T
    w = similar(x)
    if is_gpu(x)
        dwt!(x, w, b, l |> collect; wpt = wpt, convention = convention)
    else
        dwt!(StridedView(x), StridedView(w), b, l |> collect; wpt = wpt, convention = convention)
    end
end

@fastfun function idwt!(x::A, b, l :: Int; wpt = false, convention = :aligned) :: A where {T, N, A <: AbstractArray{T,N}}
    w = similar(x)
    if is_gpu(x)
        idwt!(x, w, b, repeat([l], N); wpt = wpt, convention = convention)
    else
        idwt!(StridedView(x), StridedView(w), b, repeat([l], N); wpt = wpt, convention = convention)
    end
end

@fastfun function idwt!(x::T, b, l; wpt = false, convention = :aligned) :: T where T
    w = similar(x)
    if is_gpu(x)
        idwt!(x, w, b, l |> collect; wpt = wpt, convention = convention)
    else
        idwt!(StridedView(x), StridedView(w), b, l |> collect; wpt = wpt, convention = convention)
    end
end

@fastfun function dwt(x::T, rest...; wpt = false, convention = :aligned) :: T where T
    dwt!(copy(x), rest...; wpt = wpt, convention = convention)
end

@fastfun function idwt(x::T, rest...; wpt = false, convention = :aligned) :: T where T
    idwt!(copy(x), rest...; wpt = wpt, convention = convention)
end

@fastfun wpt!(args...; convention = :aligned) = dwt!(args...; wpt=true, convention = convention)
@fastfun iwpt!(args...; convention = :aligned) = idwt!(args...; wpt=true, convention = convention)
@fastfun wpt(args...; convention = :aligned) = dwt(args...; wpt=true, convention = convention)
@fastfun iwpt(args...; convention = :aligned) = idwt(args...; wpt=true, convention = convention)
