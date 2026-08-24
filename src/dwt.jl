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

macro maybe_thread(s::Symbol, expr...)
    return quote
        if $s
            @inbounds @floop $(expr...)
        else
            @inbounds $(expr...)
        end
    end |> esc
end

@fastfun function dwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt = false,
    top = false
) :: A where {T <: Number, A <: AbstractArray{T,1}}

    is_gpu(x) ? _dwt_gpu!(x, w, b, l, wpt = wpt) : _dwt!(x, w, b, l[1], wpt = wpt)
    return x
end


@fastfun function dwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt = false,
    top = true
) :: A where {T <: Number, N, A <: AbstractArray{T,N}}

    is_gpu(x) && return _dwt_gpu!(x, w, b, l, wpt = wpt)

    l1 = Int.(l[1:N-1] .>= 1)
    passed1 = any(l1 .> 0) && (N >= 3 || size(x, 1) >= 2)
    passed2 = l[end] >= 1 && size(x, N) >= 2

    # The level-1 passes are write-through (no copyto!): the data ping-pongs
    # between x and w instead of being copied back after every pass.
    # datanow == 1 => data in x, 2 => data in w.
    datanow = 1
    if passed1
        if N == 2
            @maybe_thread top for (xs, ws) in zipslices_1d(x, w)
                _dwt_level1!(xs, ws, b)
            end
            datanow = 2
        else
            @maybe_thread top for (xs, ws) in zipslices_1d(x, w)
                dwt!(xs, ws, b, l1, wpt = wpt; top=false)
            end
        end
    end

    if passed2
        if datanow == 1
            @maybe_thread top for (xs, ws) in zipslices_nd(x, w)
                _dwt_level1!(xs, ws, b)
            end
            datanow = 2
        else
            @maybe_thread top for (xs, ws) in zipslices_nd(w, x)
                _dwt_level1!(xs, ws, b)
            end
            datanow = 1
        end
    end

    lm1 = l .- 1
    if any(lm1 .> 0)
        if datanow == 1
            @maybe_thread top for subspace in subspaces(w, x, wpt)
                wss, xss = subspace
                dwt!(xss, wss, b, lm1, wpt = wpt, top=false)
            end
        else
            @maybe_thread top for subspace in subspaces(x, w, wpt)
                wss, xss = subspace
                dwt!(xss, wss, b, lm1, wpt = wpt, top=false)
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
    top=false
) :: A where {T <: Number, A <: AbstractArray{T,1}}

    is_gpu(x) && return _idwt_gpu!(x, w, b, l, wpt = wpt)
    return _idwt!(x, w, b, l[1], wpt=wpt)
end

@fastfun function idwt!(
    x :: A,
    w :: A,
    b :: WTOrthogonalBasis,
    l :: Vector;
    wpt=false,
    top=true
) :: A where {T <: Number, N, A <: AbstractArray{T,N}}

    is_gpu(x) && return _idwt_gpu!(x, w, b, l, wpt = wpt)

    lm1 = l .- 1
    any(lm1 .> 0) && @maybe_thread top for subspace in subspaces(w, x, wpt)
        wss, xss = subspace
        idwt!(xss, wss, b, lm1, wpt = wpt, top=false)
    end

    lc = Int.(l .>= 1)
    passed2 = lc[end] >= 1 && size(x, N) >= 2
    l1 = lc[1:N-1]
    passed1 = any(l1 .> 0) && (N >= 3 || size(x, 1) >= 2)

    # Data is in x after the recursion; the level-1 passes are write-through.
    # datanow == 1 => data in x, 2 => data in w.
    datanow = 1
    if passed2
        @maybe_thread top for (xs, ws) in zipslices_nd(x, w)
            _idwt_level1!(xs, ws, b)
        end
        datanow = 2
    end

    if passed1
        if N == 2
            if datanow == 1
                @maybe_thread top for (xs, ws) in zipslices_1d(x, w)
                    _idwt_level1!(xs, ws, b)
                end
                datanow = 2
            else
                @maybe_thread top for (xs, ws) in zipslices_1d(w, x)
                    _idwt_level1!(xs, ws, b)
                end
                datanow = 1
            end
        elseif datanow == 1
            @maybe_thread top for (xs, ws) in zipslices_1d(x, w)
                idwt!(xs, ws, b, l1, wpt = wpt, top=false)
            end
        else
            @maybe_thread top for (xs, ws) in zipslices_1d(w, x)
                idwt!(xs, ws, b, l1, wpt = wpt, top=false)
            end
        end
    end

    datanow == 2 && copyto!(x, w)
    return x
end

@fastfun function dwt!(x::A, b, l :: Int; wpt = false) :: A where {T, N, A <: AbstractArray{T,N}}
    w = similar(x)
    if is_gpu(x)
        dwt!(x, w, b, repeat([l], N); wpt = wpt)
    else
        dwt!(StridedView(x), StridedView(w), b, repeat([l], N); wpt = wpt)
    end
end

@fastfun function dwt!(x::T, b, l; wpt = false) :: T where T
    w = similar(x)
    if is_gpu(x)
        dwt!(x, w, b, l |> collect; wpt = wpt)
    else
        dwt!(StridedView(x), StridedView(w), b, l |> collect; wpt = wpt)
    end
end

@fastfun function idwt!(x::A, b, l :: Int; wpt = false) :: A where {T, N, A <: AbstractArray{T,N}}
    w = similar(x)
    if is_gpu(x)
        idwt!(x, w, b, repeat([l], N); wpt = wpt)
    else
        idwt!(StridedView(x), StridedView(w), b, repeat([l], N); wpt = wpt)
    end
end

@fastfun function idwt!(x::T, b, l; wpt = false) :: T where T
    w = similar(x)
    if is_gpu(x)
        idwt!(x, w, b, l |> collect; wpt = wpt)
    else
        idwt!(StridedView(x), StridedView(w), b, l |> collect; wpt = wpt)
    end
end

@fastfun function dwt(x::T, rest...; wpt = false) :: T where T
    dwt!(copy(x), rest...; wpt = wpt)
end

@fastfun function idwt(x::T, rest...; wpt = false) :: T where T
    idwt!(copy(x), rest...; wpt = wpt)
end

@fastfun wpt!(args...) = dwt!(args...; wpt=true)
@fastfun iwpt!(args...) = idwt!(args...; wpt=true)
@fastfun wpt(args...) = dwt(args...; wpt=true)
@fastfun iwpt(args...) = idwt(args...; wpt=true)
