# Discrete wavelet transform, 1-D
@turbofun function _dwt!(
    x :: AbstractArray{T,1},
    w :: AbstractArray{T,1},
    b :: WTOrthogonalBasis,
    level=1;
    wpt=false
) :: AbstractArray{T,1} where T <: Number

    level <= 0 && return x

    _dwt_inner_loop!(x, w, b)
    # TODO: remove copyto!
    copyto!(x, w)
    if level > 1
        @strided _dwt!(
            x[1:end>>1],
            w[1:end>>1],
            b, level-1;
            wpt=wpt,
        )
        @strided wpt && _dwt!(
            x[(end>>1)+1:end],
            w[(end>>1)+1:end],
            b, level-1;
            wpt=wpt,
        )
    end

    return x
end

@turbofun function _idwt!(
    x :: AbstractArray{T, 1},
    w :: AbstractArray{T, 1},
    b :: WTOrthogonalBasis,
    level=1;
    wpt=false
) :: AbstractArray{T,1} where T <: Number

    level <= 0 && return x
    m = length(x) >> 1

    if level > 1
        _idwt!(
            x[1:m],
            w[1:m],
            b, level - 1, wpt=wpt
        )
        wpt && _idwt!(
            x[m+1:end],
            w[m+1:end],
            b, level - 1, wpt=wpt
        )
    end

    w .= zero(T)

    _idwt_inner_loop!(x, ls, hs, w, b)

    copyto!(x, w)
    return x
end

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

    # N.B.: memory copy is good for performance here.
    # 3x speed improvement just by not using views.
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

#     # => dispatch 1-D transforms over all N-1 dimensional indices.


function nsdwt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l :: Int;
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}
    w = similar(x)
    nsdwt!(x, w, b, Tuple(l for _ in 1:N), wpt=wpt)
end

function nsidwt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l :: Int;
    wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}
    w = similar(x)
    nsidwt!(x, w, b, Tuple(l for _ in 1:N), wpt=wpt)
end

function nsdwt(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l; wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}
    nsdwt!(copy(x), similar(x), b, l, wpt=wpt)
end

function nsidwt(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l; wpt = false
) :: AbstractArray{T,N} where {T <: Number, N}
    nsidwt!(copy(x), similar(x), b, l, wpt=wpt)
end

function nswpt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l;
) :: AbstractArray{T,N} where {T <: Number, N}
    nsdwt!(x, b, l, wpt=true)
end

function nsiwpt!(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l;
) :: AbstractArray{T,N} where {T <: Number, N}
    nsidwt!(x, b, l, wpt=true)
end

function nswpt(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l;
) :: AbstractArray{T,N} where {T <: Number, N}
    nsdwt!(copy(x), similar(x), b, l, wpt=true)
end

function nsiwpt(
    x :: AbstractArray{T,N},
    b :: WTOrthogonalBasis,
    l;
) :: AbstractArray{T,N} where {T <: Number, N}
    nsidwt!(copy(x), similar(x), b, l, wpt=true)
end
