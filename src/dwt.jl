# TODO:
# -> mutate the transform level in-place.
# -> branch more intelligently into the 1-D case.

function dwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    l :: NTuple{1, Int};
    wpt = false
) :: AbstractArray{T,1} where {T <: Number}

    _dwt!(x, w, b, l[1], wpt=wpt)
    return x
end

function dwt!(
    x :: AbstractArray{T,N},
    w :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}

    s = size(x, N)
    level = level |> collect

    for j=1:maximum(level)
        @floop for i=1:s
            @strided dwt!(
                x[.., i],
                w[.., i], b,
                Tuple(l >= 1 ? 1 : l for l in level[1:end-1]),
                wpt = wpt
            )
        end

        # N.B.: memory copy is good for performance here.
        # 3x speed improvement just by not using views.
        @floop for i in product([1:size(x)[l] for l=1:N-1]...)
            level[end] >= 1 && @strided _dwt!(
                x[i..., :],
                w[i..., :],
                b, wpt = wpt,
            )
        end
        
        level .-= 1
    end

    return x
end

function idwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    level :: NTuple{1, Int};
    wpt=false
) :: AbstractArray{T,1} where {T <: Number}

    return _idwt!(x, w, b, level[1], wpt=wpt)
end

function idwt!(
    x :: AbstractArray{T, N},
    w :: AbstractArray{T, N},
    b :: WTOrthogonalBasis,
    level :: NTuple{N, Int};
    wpt=false
) :: AbstractArray{T,N} where {T <: Number, N}


    s = size(x, N)

    for j=1:maximum(level)
        @floop for i in product([1:size(x)[l] for l=1:N-1]...)
            level[end] >= 1 && @strided _idwt!(
                x[i..., :],
                w[i..., :],
                b, wpt = wpt,
            )
        end

        @threads for i=1:s
            @strided idwt!(
                x[.., i],
                w[.., i], b,
                Tuple(l >= 1 ? 1 : l for l in level[1:end-1]),
                wpt = wpt
            )
        end
    end

    return x
end

dwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false) where {T,N} =
    dwt!(x, similar(x), b, Tuple(l for _ in 1:N); wpt = wpt)

dwt!(x::AbstractArray{T,N}, b, l; wpt = false) where {T,N} =
    dwt!(x, similar(x), b, l; wpt = wpt)

idwt!(x::AbstractArray{T,N}, b, l :: Int; wpt = false) where {T,N} =
    idwt!(x, similar(x), b, Tuple(l for _ in 1:N); wpt = wpt)

idwt!(x::AbstractArray{T,N}, b, l; wpt = false) where {T,N} =
    idwt!(x, similar(x), b, l; wpt = wpt)

dwt(x, rest...; wpt = false) =
    dwt!(copy(x), rest...; wpt = wpt)
idwt(x, rest...; wpt = false) =
    idwt!(copy(x), rest...; wpt = wpt)

wpt!(args...) = dwt!(args...; wpt=true)
iwpt!(args...) = idwt(args...; wpt=true)
