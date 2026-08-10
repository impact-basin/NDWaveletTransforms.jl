function nsdwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    level :: NTuple{1, Int};
    wpt = false
) :: AbstractArray{T,1} where {T <: Number}

    _dwt!(x, w, b, level[1], wpt=wpt,)
    return x
end

function nsdwt!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}

    s = size(x, N)

    @floop for i=1:s
        @strided nsdwt!(
            x[.., i],
            w[.., i],
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

function nsidwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    level :: NTuple{1, Int};
    wpt=false
) :: AbstractArray{T,1} where {T <: Number}

    return _idwt!(x, w, b, level[1], wpt=wpt)
end

function nsidwt!(
    x :: AbstractArray{T, N},
    w :: AbstractArray{T, N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt=false
) :: AbstractArray{T,N} where {T <: Number, N}


    s = size(x, N)

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
            x[.., i],
            w[.., i],
            b, level[1:end-1],
            wpt = wpt
        )
    end

    return x
end

nsdwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false) where {T,N} =
    nsdwt!(x, similar(x), b, Tuple(l for _ in 1:N); wpt = wpt)

nsdwt!(x::AbstractArray{T,N}, b, l; wpt = false) where {T,N} =
    nsdwt!(x, similar(x), b, l; wpt = wpt)

nsidwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false) where {T,N} =
    nsidwt!(x, similar(x), b, Tuple(l for _ in 1:N); wpt = wpt)

nsidwt!(x::AbstractArray{T,N}, b, l; wpt = false) where {T,N} =
    nsidwt!(x, similar(x), b, l; wpt = wpt)

nsdwt(x, rest...; wpt = false) =
    nsdwt!(copy(x), rest...; wpt = wpt)
nsidwt(x, rest...; wpt = false) =
    nsidwt!(copy(x), rest...; wpt = wpt)

nswpt!(args...) = nsdwt!(args...; wpt=true)
nsiwpt!(args...) = nsidwt(args...; wpt=true)
