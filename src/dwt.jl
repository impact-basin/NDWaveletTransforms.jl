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
