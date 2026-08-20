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


@fastfun function dwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt = false
) :: AbstractArray{T,1} where {T <: Number}

    _dwt!(x, w, b, l[1], wpt=wpt)
    return x
end

@fastfun function dwt!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}

    s = size(x, N)

    @floop for (xs, ws) in zipslices_1d(x, w)
        dwt!(xs, ws, b, [ll >= 1 for ll in l[1:N-1]], wpt = wpt)
    end

    l[end] >= 1 && @floop for (xs, ws) in zipslices_nd(x, w)
        _dwt!(xs, ws, b, 1, wpt = wpt)
    end

    l .-= 1
    any(l .> 0) && @floop for subspace in subspaces(w, x, wpt)
        wss, xss = subspace
        dwt!(xss, wss, b, l, wpt = wpt)
    end

    return x
end

@fastfun function idwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt=false
) :: AbstractArray{T,1} where {T <: Number}

    return _idwt!(x, w, b, l[1], wpt=wpt)
end

@fastfun function idwt!(
    x :: AbstractArray{T, N},
    w :: AbstractArray{T, N},
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt=false
) :: AbstractArray{T,N} where {T <: Number, N}


    any(l .> 1) && @floop for subspace in subspaces(w, x, wpt)
        wss, xss = subspace
        idwt!(xss, wss, b, l .- 1, wpt = wpt)
    end

    l .= l .>= 1

    l[end] >= 1 && @floop for (xs, ws) in zipslices_nd(x,w)
        _idwt!(xs, ws, b, 1, wpt = wpt)
    end

    @floop for (xs, ws) in zipslices_1d(x, w)
        idwt!(xs, ws, b, l[1:N-1], wpt = wpt)
    end

    return x
end

@fastfun dwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false) where {T,N} =
    dwt!(StridedView(x), StridedView(similar(x)), b, repeat([l], N); wpt = wpt)

@fastfun dwt!(x::AbstractArray{T,N}, b, l; wpt = false) where {T,N} =
    dwt!(StridedView(x), StridedView(similar(x)), b, l |> collect; wpt = wpt)

@fastfun idwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false) where {T,N} =
    idwt!(StridedView(x), StridedView(similar(x)), b, repeat([l], N); wpt = wpt)

@fastfun idwt!(x::AbstractArray{T,N}, b, l; wpt = false) where {T,N} =
    idwt!(StridedView(x), StridedView(similar(x)), b, l |> collect; wpt = wpt)

@fastfun dwt(x::T, rest...; wpt = false) where T =
    dwt!(copy(x), rest...; wpt = wpt) |> T
@fastfun idwt(x::T, rest...; wpt = false) where T =
    idwt!(copy(x), rest...; wpt = wpt) |> T

@fastfun wpt!(args...) = dwt!(args...; wpt=true)
@fastfun iwpt!(args...) = idwt!(args...; wpt=true)
@fastfun wpt(args...) = dwt(args...; wpt=true)
@fastfun iwpt(args...) = idwt(args...; wpt=true)
