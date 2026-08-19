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

    @floop for (xs, ws) in zip(
        eachslice(x, dims = N),
        eachslice(w, dims = N),
    )
        dwt!(xs, ws, b, [ll >= 1 for ll in l[1:N-1]], wpt = wpt)
    end

    l[end] >= 1 && @floop for (xs, ws) in zip(
        eachslice(x, dims = ntuple(i -> i, N - 1)),
        eachslice(w, dims = ntuple(i -> i, N - 1)),
    )
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

    l[end] >= 1 && @floop for (xs, ws) in zip(
        eachslice(x, dims = ntuple(i -> i, N - 1)),
        eachslice(w, dims = ntuple(i -> i, N - 1)),
    )
        _idwt!(xs, ws, b, 1, wpt = wpt)
    end

    @floop for (xs, ws) in zip(
        eachslice(x, dims = N),
        eachslice(w, dims = N),
    )
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
