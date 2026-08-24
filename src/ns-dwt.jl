@fastfun function nsdwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    level :: NTuple{1, Int};
    wpt = false
) :: AbstractArray{T,1} where {T <: Number}

    is_gpu(x) && return _nsdwt_gpu!(x, w, b, level, wpt = wpt)
    _dwt!(x, w, b, level[1], wpt=wpt,)
    return x
end

@fastfun function nsdwt!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}

    is_gpu(x) && return _nsdwt_gpu!(x, w, b, level, wpt = wpt)

    s = size(x, N)
    dims = ntuple(_ -> Colon(), N-1)

    @floop for i=1:s
        @strided nsdwt!(
            view(x, dims..., i),
            view(w, dims..., i),
            b, level[1:end-1],
            wpt = wpt
        )
    end

    @floop for i in product([1:size(x)[l] for l=1:N-1]...)
        @strided _dwt!(
            x[i..., :],
            w[i..., :],
            b, level[end],
            wpt = wpt,
        )
    end

    return x
end

@fastfun function nsidwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    level :: NTuple{1, Int};
    wpt=false
) :: AbstractArray{T,1} where {T <: Number}

    is_gpu(x) && return _nsidwt_gpu!(x, w, b, level, wpt = wpt)
    return _idwt!(x, w, b, level[1], wpt=wpt)
end

@fastfun function nsidwt!(
    x :: AbstractArray{T, N},
    w :: AbstractArray{T, N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt=false
) :: AbstractArray{T,N} where {T <: Number, N}


    is_gpu(x) && return _nsidwt_gpu!(x, w, b, level, wpt = wpt)

    s = size(x, N)
    dims = ntuple(_ -> Colon(), N-1)

    @floop for i in product([1:size(x)[l] for l=1:N-1]...)
         @strided _idwt!(
            x[i..., :],
            w[i..., :],
            b, level[end],
            wpt = wpt,
        )
    end

    @threads for i=1:s
        @strided nsidwt!(
            view(x, dims..., i),
            view(w, dims..., i),
            b, level[1:end-1],
            wpt = wpt
        )
    end

    return x
end

@fastfun nsdwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false) where {T,N} =
    nsdwt!(x, similar(x), b, Tuple(l for _ in 1:N); wpt = wpt)

@fastfun nsdwt!(x::AbstractArray{T,N}, b, l; wpt = false) where {T,N} =
    nsdwt!(x, similar(x), b, l; wpt = wpt)

@fastfun nsidwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false) where {T,N} =
    nsidwt!(x, similar(x), b, Tuple(l for _ in 1:N); wpt = wpt)

@fastfun nsidwt!(x::AbstractArray{T,N}, b, l; wpt = false) where {T,N} =
    nsidwt!(x, similar(x), b, l; wpt = wpt)

@fastfun nsdwt(x, rest...; wpt = false) =
    nsdwt!(copy(x), rest...; wpt = wpt)
@fastfun nsidwt(x, rest...; wpt = false) =
    nsidwt!(copy(x), rest...; wpt = wpt)

@fastfun nswpt!(args...) = nsdwt!(args...; wpt=true)
@fastfun nsiwpt!(args...) = nsidwt!(args...; wpt=true)
@fastfun nswpt(args...) = nsdwt(args...; wpt=true)
@fastfun nsiwpt(args...) = nsidwt(args...; wpt=true)
