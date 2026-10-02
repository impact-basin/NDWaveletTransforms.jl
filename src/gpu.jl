# ------------------------------------------------------------------
# Forward transform along one axis, one level, over every line in the
# region: each thread computes one (low, high) output pair of one line.
# Aligned phase (default): both filters act on the same window.
# ------------------------------------------------------------------
@kernel function _ka_dwt_axis!(src, dst, φ, ψ, off, region, fstrides, ax, ntaps)
    i = @index(Global)
    nactive = region[ax]
    npairs = nactive >> 1
    p = mod1(i, npairs)                 # 1-based pair within the line
    slice = fld(i - 1, npairs)          # 0-based line index
    base = off
    rem = slice
    for d in 1:length(region)
        if d != ax
            c = mod(rem, region[d]) + 1
            base += (c - 1) * fstrides[d]
            rem = fld(rem, region[d])
        end
    end
    sa = fstrides[ax]
    sigi = 2p - 1
    sφ = zero(eltype(src))
    sψ = zero(eltype(src))
    @inbounds for k in 1:ntaps
        xv = src[base + (mod1(sigi + k - 1, nactive) - 1) * sa]
        sφ += φ[k] * xv
        sψ += ψ[k] * xv
    end
    dst[base + (p - 1) * sa] = sφ
    dst[base + (npairs + p - 1) * sa] = sψ
end

# Wavelets.jl-compatible phase: the wavelet filter's window starts ntaps-2
# taps before the scaling filter's window (d[j] = Σ ψ[k] x[2j+k-ntaps]).
@kernel function _ka_dwt_axis_wv!(src, dst, φ, ψ, off, region, fstrides, ax, ntaps)
    i = @index(Global)
    nactive = region[ax]
    npairs = nactive >> 1
    p = mod1(i, npairs)                 # 1-based pair within the line
    slice = fld(i - 1, npairs)          # 0-based line index
    base = off
    rem = slice
    for d in 1:length(region)
        if d != ax
            c = mod(rem, region[d]) + 1
            base += (c - 1) * fstrides[d]
            rem = fld(rem, region[d])
        end
    end
    sa = fstrides[ax]
    sigi = 2p - 1
    sφ = zero(eltype(src))
    sψ = zero(eltype(src))
    @inbounds for k in 1:ntaps
        xv = src[base + (mod1(sigi + k - 1, nactive) - 1) * sa]
        sφ += φ[k] * xv
        xvψ = src[base + (mod1(sigi - ntaps + 2 + k - 1, nactive) - 1) * sa]
        sψ += ψ[k] * xvψ
    end
    dst[base + (p - 1) * sa] = sφ
    dst[base + (npairs + p - 1) * sa] = sψ
end

# ----------------------------------------------
# Inverse (synthesis) along one axis, one level.
# Aligned phase: scaling and wavelet coefficients
# feed the same window per output position.
# ----------------------------------------------
@kernel function _ka_idwt_axis!(src, hsrc, dst, φ, ψ, off, region, fstrides, ax, ntaps)
    i = @index(Global)
    nactive = region[ax]
    m = nactive >> 1
    pos = mod1(i, nactive)              # 1-based output position in the line
    slice = fld(i - 1, nactive)
    base = off
    rem = slice
    for d in 1:length(region)
        if d != ax
            c = mod(rem, region[d]) + 1
            base += (c - 1) * fstrides[d]
            rem = fld(rem, region[d])
        end
    end
    sa = fstrides[ax]
    acc = zero(eltype(dst))
    # every (i2, k) with mod1(2i2-1+k-1, nactive) == pos, i.e.
    # k = pos + t*nactive - 2(i2-1) for t = 0, 1, ...
    tmax_t = (ntaps + nactive - 1) ÷ nactive
    for t in 0:tmax_t
        lo = cld(pos + t * nactive - ntaps + 2, 2)
        hi = fld(pos + t * nactive + 1, 2)
        lo = max(1, lo)
        hi = min(m, hi)
        @inbounds for i2 in lo:hi
            k = pos + t * nactive - 2 * (i2 - 1)
            acc += src[base + (i2 - 1) * sa] * φ[k] +
                   hsrc[base + (m + i2 - 1) * sa] * ψ[k]
        end
    end
    dst[base + (pos - 1) * sa] = acc
end

# Wavelets.jl-compatible phase: scaling coefficients feed the window
# [2i2-1, 2i2+ntaps-2] (k = pos + t*nactive - 2(i2-1)) while wavelet
# coefficients feed the window [2i2-ntaps+1, 2i2]
# (k = pos + t*nactive - 2*i2 + ntaps).
@kernel function _ka_idwt_axis_wv!(src, hsrc, dst, φ, ψ, off, region, fstrides, ax, ntaps)
    i = @index(Global)
    nactive = region[ax]
    m = nactive >> 1
    pos = mod1(i, nactive)              # 1-based output position in the line
    slice = fld(i - 1, nactive)
    base = off
    rem = slice
    for d in 1:length(region)
        if d != ax
            c = mod(rem, region[d]) + 1
            base += (c - 1) * fstrides[d]
            rem = fld(rem, region[d])
        end
    end
    sa = fstrides[ax]
    acc = zero(eltype(dst))
    tmax_t = (ntaps + nactive - 1) ÷ nactive
    # scaling coefficients: mod1(2i2-1+k-1, nactive) == pos
    for t in 0:tmax_t
        lo = cld(pos + t * nactive - ntaps + 2, 2)
        hi = fld(pos + t * nactive + 1, 2)
        lo = max(1, lo)
        hi = min(m, hi)
        @inbounds for i2 in lo:hi
            k = pos + t * nactive - 2 * (i2 - 1)
            acc += src[base + (i2 - 1) * sa] * φ[k]
        end
    end
    # wavelet coefficients: mod1(2i2-ntaps+k, nactive) == pos, i.e.
    # k = pos + t*nactive - 2*i2 + ntaps for integer t. Unlike the scaling
    # window, this window starts before the pair, so t can be negative when
    # the block is small (the window wraps more than once).
    tmin = cld(3 - pos - ntaps, nactive)
    tmax = fld(nactive + 1 - pos, nactive)
    for t in tmin:tmax
        lo = cld(pos + t * nactive, 2)
        hi = fld(pos + t * nactive + ntaps - 1, 2)
        lo = max(1, lo)
        hi = min(m, hi)
        @inbounds for i2 in lo:hi
            k = pos + t * nactive - 2 * i2 + ntaps
            acc += hsrc[base + (m + i2 - 1) * sa] * ψ[k]
        end
    end
    dst[base + (pos - 1) * sa] = acc
end

# ------------------------------------------------------------------
# Host-side helpers
# ------------------------------------------------------------------

# 1-based linear offset of block `blk` at level `lv` (block index has one
# 0-based coordinate per dim, radix 2^(lv-1)).
@inline function _gpu_block_offset(blk, bsize, fstrides, N, lv)
    off = 1
    rem = blk
    nbd = 1 << (lv - 1)
    for d in 1:N
        bd = mod(rem, nbd)
        off += bd * bsize[d] * fstrides[d]
        rem = fld(rem, nbd)
    end
    off
end

# Per-dim 0-based block coordinates of block `blk` at level `lv`.
@inline function _gpu_block_coords(blk, N, lv)
    coords = ntuple(_ -> 0, N)
    rem = blk
    nbd = 1 << (lv - 1)
    for d in 1:N
        coords = Base.setindex(coords, mod(rem, nbd), d)
        rem = fld(rem, nbd)
    end
    coords
end

@inline _gpu_fstrides(n) = ntuple(d -> prod(n[1:d-1]), length(n))
@inline _gpu_nslices(region, ax) = prod(region) ÷ region[ax]

@inline function _gpu_coeffs(backend, b, T)
    (Adapt.adapt(backend, collect(T.(b.φ))),
     Adapt.adapt(backend, collect(T.(b.ψ))))
end

@inline function _gpu_launch(kernel, backend, nthreads, args...)
    kernel(backend, min(256, nthreads))(args..., ndrange=nthreads)
    nothing
end

function _gpu_dwt_axis!(backend, src, dst, φd, ψd, off, region, fstrides, d, ntaps, ::Val{:aligned})
    nthreads = (region[d] >> 1) * _gpu_nslices(region, d)
    nthreads < 1 && return nothing
    _gpu_launch(_ka_dwt_axis!, backend, nthreads,
                src, dst, φd, ψd, off, region, fstrides, d, ntaps)
end

function _gpu_dwt_axis!(backend, src, dst, φd, ψd, off, region, fstrides, d, ntaps, ::Val{:wavelets})
    nthreads = (region[d] >> 1) * _gpu_nslices(region, d)
    nthreads < 1 && return nothing
    _gpu_launch(_ka_dwt_axis_wv!, backend, nthreads,
                src, dst, φd, ψd, off, region, fstrides, d, ntaps)
end

function _gpu_idwt_axis!(backend, src, dst, φd, ψd, off, region, fstrides, d, ntaps, ::Val{:aligned})
    nthreads = region[d] * _gpu_nslices(region, d)
    nthreads < 1 && return nothing
    # The drivers keep the data coherent in `src` (both halves), so the
    # kernel reads the detail part from the same buffer and no stash/race
    # handling is needed.
    _gpu_launch(_ka_idwt_axis!, backend, nthreads,
                src, src, dst, φd, ψd, off, region, fstrides, d, ntaps)
end

function _gpu_idwt_axis!(backend, src, dst, φd, ψd, off, region, fstrides, d, ntaps, ::Val{:wavelets})
    nthreads = region[d] * _gpu_nslices(region, d)
    nthreads < 1 && return nothing
    _gpu_launch(_ka_idwt_axis_wv!, backend, nthreads,
                src, src, dst, φd, ψd, off, region, fstrides, d, ntaps)
end

# ------------------------------------------------------------------
# forward / inverse N-D transforms on device arrays.
# l is a Vector of per-dimension levels (mirrors the CPU entry points).
#
# The level-1 passes ping-pong between x and w. A pass writes the
# transform of its region into the output buffer; the final value of a
# position is wherever the last pass covering it wrote it. For the forward
# transform that leaves some final pieces stranded in w: the ring of each
# level whose passes ended in w, relative to the next level's coverage.
# Those rings are gathered back into x afterwards. (For the inverse the
# last pass always covers the full array, so a single final copy suffices.)
# ------------------------------------------------------------------

# Gather the forward's w-stranded rings.
function _gpu_gather!(x, w, levels, wpt, N)
    for i in eachindex(levels)
        dn, R = levels[i]
        dn == 2 || continue
        islast = i == length(levels)
        Rn = islast ? ntuple(_ -> 0, N) : levels[i+1][2]
        nblk = wpt ? (1 << (N * (i - 1))) : 1
        for blk in 0:nblk-1
            c = _gpu_block_coords(blk, N, i)
            if islast
                ranges = ntuple(k -> (c[k] * R[k] + 1):(c[k] * R[k] + R[k]), N)
                copyto!(view(x, ranges...), view(w, ranges...))
            elseif !wpt
                # ring of the block relative to the next level's region
                for k in 1:N
                    ranges = ntuple(j -> begin
                        b0 = c[j] * R[j]
                        if j == k
                            (b0 + Rn[j] + 1):(b0 + R[j])
                        else
                            (b0 + 1):(b0 + R[j])
                        end
                    end, N)
                    copyto!(view(x, ranges...), view(w, ranges...))
                end
            end
        end
    end
    nothing
end

function _dwt_gpu!(x, w, b, l::Vector, ::Val{C}; wpt=false) where {C}
    N = ndims(x)
    n = size(x)
    backend = KernelAbstractions.get_backend(x)
    φd, ψd = _gpu_coeffs(backend, b, eltype(x))
    fstrides = _gpu_fstrides(n)
    ntaps = length(b.φ)

    datanow = 1                      # 1 => data in x, 2 => data in w
    levels = Tuple{Int,NTuple{N,Int}}[]
    lcur = collect(Int, l)
    lv = 1
    while any(lcur .> 0)
        bsize = ntuple(d -> n[d] >> (lv - 1), N)
        all(bsize .< 2) && break
        nblk = wpt ? (1 << (N * (lv - 1))) : 1
        ran = false
        for d in 1:N
            (lcur[d] >= 1 && bsize[d] >= 2) || continue
            ran = true
            if datanow == 1
                for blk in 0:nblk-1
                    off = _gpu_block_offset(blk, bsize, fstrides, N, lv)
                    _gpu_dwt_axis!(backend, x, w, φd, ψd, off, bsize, fstrides, d, ntaps, Val{C}())
                end
                datanow = 2
            else
                for blk in 0:nblk-1
                    off = _gpu_block_offset(blk, bsize, fstrides, N, lv)
                    _gpu_dwt_axis!(backend, w, x, φd, ψd, off, bsize, fstrides, d, ntaps, Val{C}())
                end
                datanow = 1
            end
        end
        ran && push!(levels, (datanow, bsize))
        lcur .-= 1
        lv += 1
    end
    _gpu_gather!(x, w, levels, wpt, N)
    return x
end

function _idwt_gpu!(x, w, b, l::Vector, ::Val{C}; wpt=false) where {C}
    N = ndims(x)
    n = size(x)
    backend = KernelAbstractions.get_backend(x)
    φd, ψd = _gpu_coeffs(backend, b, eltype(x))
    fstrides = _gpu_fstrides(n)
    ntaps = length(b.φ)

    # Each pass reads the (coherent) data from x and writes into w; the
    # pass's blocks are then folded back into x so the next pass always sees
    # a self-consistent array.
    for lv in maximum(l):-1:1
        bsize = ntuple(d -> n[d] >> (lv - 1), N)
        all(bsize .< 2) && continue
        nblk = wpt ? (1 << (N * (lv - 1))) : 1
        for d in N:-1:1
            (l[d] >= lv && bsize[d] >= 2) || continue
            for blk in 0:nblk-1
                off = _gpu_block_offset(blk, bsize, fstrides, N, lv)
                _gpu_idwt_axis!(backend, x, w, φd, ψd, off, bsize, fstrides, d, ntaps, Val{C}())
            end
            for blk in 0:nblk-1
                c = _gpu_block_coords(blk, N, lv)
                ranges = ntuple(k -> (c[k] * bsize[k] + 1):(c[k] * bsize[k] + bsize[k]), N)
                copyto!(view(x, ranges...), view(w, ranges...))
            end
        end
    end
    return x
end

# --------------------
# Nonstandard variants
# --------------------
function _nsdwt_gpu!(x, w, b, level, ::Val{C}; wpt=false) where {C}
    N = ndims(x)
    n = size(x)
    backend = KernelAbstractions.get_backend(x)
    φd, ψd = _gpu_coeffs(backend, b, eltype(x))
    fstrides = _gpu_fstrides(n)
    ntaps = length(b.φ)

    for d in 1:N
        for lv in 1:level[d]
            bsize = ntuple(j -> j == d ? (n[d] >> (lv - 1)) : n[j], N)
            bsize[d] < 2 && break
            nblk = wpt ? (1 << (lv - 1)) : 1
            for blk in 0:nblk-1
                off = 1 + blk * bsize[d] * fstrides[d]
                _gpu_dwt_axis!(backend, x, w, φd, ψd, off, bsize, fstrides, d, ntaps, Val{C}())
            end
            # fold this level's slabs back into x so the next axis reads a
            # self-consistent array (the axis-1 high parts would otherwise
            # stay stranded in w)
            for blk in 0:nblk-1
                c0 = blk * bsize[d]
                ranges = ntuple(j -> begin
                    if j == d
                        (c0 + 1):(c0 + bsize[j])
                    else
                        1:bsize[j]
                    end
                end, N)
                copyto!(view(x, ranges...), view(w, ranges...))
            end
        end
    end
    return x
end

function _nsidwt_gpu!(x, w, b, level, ::Val{C}; wpt=false) where {C}
    N = ndims(x)
    n = size(x)
    backend = KernelAbstractions.get_backend(x)
    φd, ψd = _gpu_coeffs(backend, b, eltype(x))
    fstrides = _gpu_fstrides(n)
    ntaps = length(b.φ)

    for d in N:-1:1
        for lv in level[d]:-1:1
            bsize = ntuple(j -> j == d ? (n[d] >> (lv - 1)) : n[j], N)
            bsize[d] < 2 && continue
            nblk = wpt ? (1 << (lv - 1)) : 1
            for blk in 0:nblk-1
                off = 1 + blk * bsize[d] * fstrides[d]
                _gpu_idwt_axis!(backend, x, w, φd, ψd, off, bsize, fstrides, d, ntaps, Val{C}())
            end
            # fold this level's slabs back into x
            for blk in 0:nblk-1
                c0 = blk * bsize[d]
                ranges = ntuple(j -> begin
                    if j == d
                        (c0 + 1):(c0 + bsize[j])
                    else
                        1:bsize[j]
                    end
                end, N)
                copyto!(view(x, ranges...), view(w, ranges...))
            end
        end
    end
    return x
end

@inline is_gpu(x) = x isa AbstractGPUArray
