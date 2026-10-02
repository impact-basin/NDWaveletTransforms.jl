@fastfun function nsdwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    level :: NTuple{1, Int};
    wpt = false,
    convention = :aligned
) :: AbstractArray{T,1} where {T <: Number}

    is_gpu(x) && return _nsdwt_gpu!(x, w, b, level, Val(convention), wpt = wpt)
    _dwt!(x, w, b, level[1], Val(convention), wpt=wpt,)
    return x
end

@fastfun function nsdwt!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt = false,
    convention = :aligned
) :: AbstractArray{T,N} where {T <: Number, N}

    is_gpu(x) && return _nsdwt_gpu!(x, w, b, level, Val(convention), wpt = wpt)

    s = size(x, N)
    dims = ntuple(_ -> Colon(), N-1)
    # The nonstandard driver threads whole slice loops, so the relevant
    # work is the whole loop, not a single pass; a much lower threshold
    # applies than for the fused level-1 passes.
    dothread = Threads.nthreads() > 1 && length(x) * length(b.φ) >= 8_000

    if dothread
        @floop for i=1:s
            @strided nsdwt!(
                view(x, dims..., i),
                view(w, dims..., i),
                b, level[1:end-1],
                wpt = wpt,
                convention = convention
            )
        end
    else
        for i=1:s
            @strided nsdwt!(
                view(x, dims..., i),
                view(w, dims..., i),
                b, level[1:end-1],
                wpt = wpt,
                convention = convention
            )
        end
    end

    if dothread
        @floop for i in product([1:size(x)[l] for l=1:N-1]...)
            @strided _dwt!(
                x[i..., :],
                w[i..., :],
                b, level[end],
                Val(convention),
                wpt = wpt,
            )
        end
    else
        for i in product([1:size(x)[l] for l=1:N-1]...)
            @strided _dwt!(
                x[i..., :],
                w[i..., :],
                b, level[end],
                Val(convention),
                wpt = wpt,
            )
        end
    end

    return x
end

@fastfun function nsidwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    level :: NTuple{1, Int};
    wpt=false,
    convention = :aligned
) :: AbstractArray{T,1} where {T <: Number}

    is_gpu(x) && return _nsidwt_gpu!(x, w, b, level, Val(convention), wpt = wpt)
    return _idwt!(x, w, b, level[1], Val(convention), wpt=wpt)
end

@fastfun function nsidwt!(
    x :: AbstractArray{T, N},
    w :: AbstractArray{T, N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt=false,
    convention = :aligned
) :: AbstractArray{T,N} where {T <: Number, N}


    is_gpu(x) && return _nsidwt_gpu!(x, w, b, level, Val(convention), wpt = wpt)

    s = size(x, N)
    dims = ntuple(_ -> Colon(), N-1)
    # The nonstandard driver threads whole slice loops, so the relevant
    # work is the whole loop, not a single pass; a much lower threshold
    # applies than for the fused level-1 passes.
    dothread = Threads.nthreads() > 1 && length(x) * length(b.φ) >= 8_000

    if dothread
        @floop for i in product([1:size(x)[l] for l=1:N-1]...)
             @strided _idwt!(
                x[i..., :],
                w[i..., :],
                b, level[end],
                Val(convention),
                wpt = wpt,
            )
        end
    else
        for i in product([1:size(x)[l] for l=1:N-1]...)
             @strided _idwt!(
                x[i..., :],
                w[i..., :],
                b, level[end],
                Val(convention),
                wpt = wpt,
            )
        end
    end

    if dothread
        @threads for i=1:s
            @strided nsidwt!(
                view(x, dims..., i),
                view(w, dims..., i),
                b, level[1:end-1],
                wpt = wpt,
                convention = convention
            )
        end
    else
        for i=1:s
            @strided nsidwt!(
                view(x, dims..., i),
                view(w, dims..., i),
                b, level[1:end-1],
                wpt = wpt,
                convention = convention
            )
        end
    end

    return x
end

@fastfun nsdwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false, convention = :aligned) where {T,N} =
    nsdwt!(x, similar(x), b, Tuple(l for _ in 1:N); wpt = wpt, convention = convention)

@fastfun nsdwt!(x::AbstractArray{T,N}, b, l; wpt = false, convention = :aligned) where {T,N} =
    nsdwt!(x, similar(x), b, l; wpt = wpt, convention = convention)

@fastfun nsidwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false, convention = :aligned) where {T,N} =
    nsidwt!(x, similar(x), b, Tuple(l for _ in 1:N); wpt = wpt, convention = convention)

@fastfun nsidwt!(x::AbstractArray{T,N}, b, l; wpt = false, convention = :aligned) where {T,N} =
    nsidwt!(x, similar(x), b, l; wpt = wpt, convention = convention)

@fastfun nsdwt(x, rest...; wpt = false, convention = :aligned) =
    nsdwt!(copy(x), rest...; wpt = wpt, convention = convention)
@fastfun nsidwt(x, rest...; wpt = false, convention = :aligned) =
    nsidwt!(copy(x), rest...; wpt = wpt, convention = convention)

@fastfun nswpt!(args...; convention = :aligned) = nsdwt!(args...; wpt=true, convention = convention)
@fastfun nsiwpt!(args...; convention = :aligned) = nsidwt!(args...; wpt=true, convention = convention)
@fastfun nswpt(args...; convention = :aligned) = nsdwt(args...; wpt=true, convention = convention)
@fastfun nsiwpt(args...; convention = :aligned) = nsidwt(args...; wpt=true, convention = convention)
