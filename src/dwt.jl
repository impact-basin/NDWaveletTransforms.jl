# `@floop` boxes a captured variable that is assigned in a branch before the
# loop. The strided passes hoist n1, odims, odsize, ngrid and nblocks above
# the CONTIG split for this reason; a Box in the hot loop costs real time.
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

"""
    dwt!(x, b, l; convention = :aligned)
    dwt!(x, w, b, l; convention = :aligned)

Transform `x` in place with basis `b`, and return `x`.

`l` sets the number of levels. An `Int` applies the same number along every
axis; an `NTuple{N,Int}` or `Vector{Int}` sets the level along each axis
separately. A dimension of length `n` supports `l` levels of perfect
reconstruction when `2^l` divides `n`, and is transformed as far as it
divides beyond that.

The four-argument form takes `w`, a scratch array with the same size and
element type as `x`, and avoids the allocation that the three-argument form
makes.

`convention` selects the phase of the detail coefficients. `:aligned` (the
default) applies the scaling and wavelet filters to the same input window.
`:wavelets` reproduces the coefficient layout of Wavelets.jl. The scaling
coefficients are identical in both conventions.

`x` may be any strided `AbstractArray`, including a subband view from
[`rtree_view`](@ref), or a GPU array.

# Examples

```julia
using NDWaveletTransforms

x = rand(128, 128)
dwt!(copy(x), WT_D4, 2)         # two levels along both axes
dwt!(copy(x), WT_D4, (1, 3))    # one level along axis 1, three along axis 2
idwt!(dwt(x, WT_D4, 2), WT_D4, 2) ≈ x
```

See also [`dwt`](@ref), [`idwt!`](@ref), [`wpt!`](@ref), and the
[Phase conventions](@ref) page.
"""
@fastfun function dwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt = false,
    top = false,
    convention = :aligned
) :: A where {T <: Number, A <: AbstractArray{T,1}}

    size(x) == size(w) || throw(DimensionMismatch("x and w must have the same size"))
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

    size(x) == size(w) || throw(DimensionMismatch("x and w must have the same size"))
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

"""
    idwt!(x, b, l; convention = :aligned)
    idwt!(x, w, b, l; convention = :aligned)

Invert a wavelet transform in place, and return `x`. The arguments are those
of [`dwt!`](@ref); `convention` must match the one used for the forward
transform.
"""
@fastfun function idwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt=false,
    top=false,
    convention = :aligned
) :: A where {T <: Number, A <: AbstractArray{T,1}}

    size(x) == size(w) || throw(DimensionMismatch("x and w must have the same size"))
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

    size(x) == size(w) || throw(DimensionMismatch("x and w must have the same size"))
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

@fastfun function dwt!(x::A, b, l :: Int; wpt = false, convention = :aligned) :: A where {T, N, A <: AbstractArray{T,N}}
    w = similar(x)
    if is_gpu(x)
        dwt!(x, w, b, repeat([l], N); wpt = wpt, convention = convention)
    else
        dwt!(StridedView(x), StridedView(w), b, repeat([l], N); wpt = wpt, convention = convention)
    end
    return x
end

@fastfun function dwt!(x::T, b, l; wpt = false, convention = :aligned) :: T where T
    w = similar(x)
    if is_gpu(x)
        dwt!(x, w, b, l |> collect; wpt = wpt, convention = convention)
    else
        dwt!(StridedView(x), StridedView(w), b, l |> collect; wpt = wpt, convention = convention)
    end
    return x
end

@fastfun function idwt!(x::A, b, l :: Int; wpt = false, convention = :aligned) :: A where {T, N, A <: AbstractArray{T,N}}
    w = similar(x)
    if is_gpu(x)
        idwt!(x, w, b, repeat([l], N); wpt = wpt, convention = convention)
    else
        idwt!(StridedView(x), StridedView(w), b, repeat([l], N); wpt = wpt, convention = convention)
    end
    return x
end

@fastfun function idwt!(x::T, b, l; wpt = false, convention = :aligned) :: T where T
    w = similar(x)
    if is_gpu(x)
        idwt!(x, w, b, l |> collect; wpt = wpt, convention = convention)
    else
        idwt!(StridedView(x), StridedView(w), b, l |> collect; wpt = wpt, convention = convention)
    end
    return x
end

"""
    dwt(x, b, l; convention = :aligned)

Return a transformed copy of `x`. The copying form of [`dwt!`](@ref).
"""
@fastfun function dwt(x, rest...; wpt = false, convention = :aligned)
    dwt!(copy(x), rest...; wpt = wpt, convention = convention)
end

"""
    idwt(x, b, l; convention = :aligned)

Return an inverse-transformed copy of `x`. The copying form of
[`idwt!`](@ref).
"""
@fastfun function idwt(x, rest...; wpt = false, convention = :aligned)
    idwt!(copy(x), rest...; wpt = wpt, convention = convention)
end

"""
    wpt!(x, b, l; convention = :aligned)

Wavelet packet transform of `x` in place, and return `x`. Equivalent to
`dwt!(x, b, l; wpt = true)`: the recursion is applied to every subband rather
than only the approximation band.
"""
@fastfun wpt!(args...; convention = :aligned) = dwt!(args...; wpt=true, convention = convention)
"""
    iwpt!(x, b, l; convention = :aligned)

Invert a wavelet packet transform in place, and return `x`.
"""
@fastfun iwpt!(args...; convention = :aligned) = idwt!(args...; wpt=true, convention = convention)
"""
    wpt(x, b, l; convention = :aligned)

Return a packet-transformed copy of `x`. The copying form of
[`wpt!`](@ref).
"""
@fastfun wpt(args...; convention = :aligned) = dwt(args...; wpt=true, convention = convention)
"""
    iwpt(x, b, l; convention = :aligned)

Return an inverse packet-transformed copy of `x`.
"""
@fastfun iwpt(args...; convention = :aligned) = idwt(args...; wpt=true, convention = convention)
