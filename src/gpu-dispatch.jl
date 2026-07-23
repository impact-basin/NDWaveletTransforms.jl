function dwtk!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F};
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 1}}
    gpu = KernelAbstractions.get_backend(x)
    ls, hs = sbviews(w)
    dwt_1d! = _dwt_1d_kernel!(gpu)
    dwt_1d!(ls, x, b.φ, ndrange = length(x) >> 1)
    dwt_1d!(hs, x, b.ψ, ndrange = length(x) >> 1)
    nothing
end

# force revise to see this again

function dwtk!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F};
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    gpu = KernelAbstractions.get_backend(x)
    Nx, Ny = size(x)
    ll, lh, hl, hh = sbviews(w)
    dwt_2d! = _dwt_2d_kernel!(gpu)
    dwt_2d!(ll, x, b.φφ, ndrange = (Nx >> 1, Ny >> 1))
    dwt_2d!(lh, x, b.φψ, ndrange = (Nx >> 1, Ny >> 1))
    dwt_2d!(hl, x, b.ψφ, ndrange = (Nx >> 1, Ny >> 1))
    dwt_2d!(hh, x, b.ψψ, ndrange = (Nx >> 1, Ny >> 1))
    nothing
end

function dwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Int;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 1}}
    dwtk!(w, x, b)
    x .= w
    if l > 1
        for (sw, sx) in subspaces(w, x, wpt)
            dwt!(sw, sx, b, l - 1; wpt=wpt)
        end
    end
    nothing
end

function dwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Int;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    dwtk!(w, x, b)
    x .= w
    if l > 1
        for (sw, sx) in subspaces(w, x, wpt)
            dwt!(sw, sx, b, l - 1; wpt=wpt)
        end
    end
    nothing
end


function dwt!(
    x::T,
    b::WTOrthogonalBasis{N, F},
    l :: Int = 1;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 1}}
    gpux = CuArray(x)
    w = similar(gpux)
    dwt!(w, gpux, b, l; wpt = wpt)
    copyto!(x, gpux)
    nothing
end


function idwtk!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F};
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 1}}
    gpu = KernelAbstractions.get_backend(x)
    idwt_1d! = _idwt_1d_kernel!(gpu)
    ls, hs = sbviews(x)
    idwt_1d!(w, ls, hs, b.φ, b.ψ, ndrange = length(w))
    nothing
end

function idwtk!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F};
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    gpu = KernelAbstractions.get_backend(x)
    idwt_2d! = _idwt_2d_kernel!(gpu)
    ll, lh, hl, hh = sbviews(x)
    idwt_2d!(w,
        ll, lh, hl, hh,
        b.φφ, b.ψφ, b.φψ, b.ψψ,
        ndrange = size(w),
    )
    nothing
end

function idwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Int;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 1}}
    if l > 1
        for (sw, sx) in subspaces(w, x, wpt)
            idwt!(sw, sx, b, l - 1; wpt=wpt)
        end
    end
    idwtk!(w, x, b)
    x .= w
    nothing
end

function idwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Int;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    if l > 1
        for (sw, sx) in subspaces(w, x, wpt)
            idwt!(sw, sx, b, l - 1; wpt=wpt)
        end
    end
    idwtk!(w, x, b)
    x .= w
    nothing
end


function rowwise_dwtk!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F};
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    gpu = KernelAbstractions.get_backend(x)
    Nx, Ny = size(x)
    ls, hs = @wtview (w[:l_], w[:h_])
    row! = _dwt_row_kernel!(gpu)
    row!(ls, hs, x, b.φ, b.ψ, ndrange = (Nx >> 1, Ny))
    return nothing
end

function colwise_dwtk!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F};
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}

    gpu = KernelAbstractions.get_backend(x)
    Nx, Ny = size(x)
    ls, hs = @wtview (w[:_l], w[:_h])

    col! = _dwt_col_kernel!(gpu)
    col!(ls, hs, x, b.φ, b.ψ, ndrange = (Nx, Ny >> 1))

    return nothing
end

function rowwise_idwtk!(
    w::T, x::T,
    b::WTOrthogonalBasis{N, F};
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}

    gpu = KernelAbstractions.get_backend(w)
    Nx, Ny = size(w)
    ls, hs = @wtview (x[:l_], x[:h_])

    row! = _idwt_row_kernel!(gpu)
    row!(w, ls, hs, b.φ, b.ψ, ndrange = (Nx, Ny))

    return nothing
end

function colwise_idwtk!(
    w::T, x::T,
    b::WTOrthogonalBasis{N, F};
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}

    gpu = KernelAbstractions.get_backend(w)
    Nx, Ny = size(w)
    ls, hs = @wtview (x[:_l], x[:_h])

    col! = _idwt_col_kernel!(gpu)
    col!(w, ls, hs, b.φ, b.ψ, ndrange = (Nx, Ny))

    return nothing
end

function row_dwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Int;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    rowwise_dwtk!(w, x, b)
    x .= w
    if l > 1
        lw, hw = @wtview (w[:l_], w[:h_])
        lx, hx = @wtview (x[:l_], x[:h_])
        row_dwt!(lw, lx, b, l - 1; wpt=wpt)
        wpt && row_dwt!(hw, hx, b, l - 1; wpt=wpt)
        x .= w
    end
    nothing
end

function col_dwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Int;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    colwise_dwtk!(w, x, b)
    x .= w
    if l > 1
        lw, hw = @wtview (w[:_l], w[:_h])
        lx, hx = @wtview (x[:_l], x[:_h])
        col_dwt!(lw, lx, b, l - 1; wpt=wpt)
        wpt && col_dwt!(hw, hx, b, l - 1; wpt=wpt)
        x .= w
    end
    nothing
end

function dwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Tuple{Int, Int};
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    u = min(l...)
    l = l .- u
    dwt!(w, x, b, u; wpt=wpt)
    sp = subspaces(2, u)
    @match l begin
        (r, 0) => begin
            for i=1:length(sp)
                row_dwt!(wt_index(w, sp[i]),
                         wt_index(x, sp[i]),
                         b, r; wpt = wpt
                )
            end
        end
        (0, c) => begin
            for i=1:length(sp)
                col_dwt!(wt_index(w, sp[i]),
                         wt_index(x, sp[i]),
                         b, c; wpt = wpt
                )
            end
        end
    end
    # TODO: index into the subspaces...
    nothing
end

function row_idwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Int;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    if l > 1
        lw, hw = @wtview (w[:l_], w[:h_])
        lx, hx = @wtview (x[:l_], x[:h_])
        row_idwt!(lw, lx, b, l - 1; wpt=wpt)
        wpt && row_idwt!(hw, hx, b, l - 1; wpt=wpt)
        lx .= lw
        hx .= hw
    end
    rowwise_idwtk!(w, x, b)
    x .= w
    nothing
end

function col_idwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Int;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}
    if l > 1
        lw, hw = @wtview (w[:_l], w[:_h])
        lx, hx = @wtview (x[:_l], x[:_h])
        col_idwt!(lw, lx, b, l - 1; wpt=wpt)
        wpt && col_idwt!(hw, hx, b, l - 1; wpt=wpt)
        lx .= lw
        hx .= hw
    end
    colwise_idwtk!(w, x, b)
    x .= w
    nothing
end

function idwt!(
    w::T,
    x::T,
    b::WTOrthogonalBasis{N, F},
    l::Tuple{Int, Int};
    wpt = false
) :: Nothing where {N, F <: AbstractFloat, T <: AbstractArray{F, 2}}

    !wpt && @warn "anisotropic IDWT is bugged without WPT; results may be poor." wpt b l
    u     = min(l...)
    sp    = subspaces(2, u)

    @match (l .- u) begin
        (r, 0) => for i=1:length(sp)
            row_idwt!(wt_index(w, sp[i]),
                      wt_index(x, sp[i]),
                      b, r; wpt=wpt
            )
        end
        (0, c) => for i=1:length(sp)
            col_idwt!(wt_index(w, sp[i]),
                      wt_index(x, sp[i]),
                      b, c; wpt=wpt
            )
        end
    end

    idwt!(w, x, b, u; wpt=wpt)
    nothing
end

function dwt!(
    x::T,
    b::WTOrthogonalBasis{N, F},
    l = 1;
    wpt = false
) :: Nothing where {N, M, F <: AbstractFloat, T <: AbstractArray{F, M}}
    gpux = CuArray(x)
    w = similar(gpux)
    dwt!(w, gpux, b, l; wpt = wpt)
    copyto!(x, gpux)
    nothing
end

function dwt!(
    x::CuArray,
    b::WTOrthogonalBasis{N, F},
    l = 1;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat}
    w = similar(x)
    dwt!(w, x, b, l; wpt = wpt)
    nothing
end

function idwt!(
    x::T,
    b::WTOrthogonalBasis{N, F},
    l = 1;
    wpt = false
) :: Nothing where {N, M, F <: AbstractFloat, T <: AbstractArray{F, M}}
    gpux = CuArray(x)
    w = similar(gpux)
    idwt!(w, gpux, b, l; wpt = wpt)
    copyto!(x, gpux)
    nothing
end

function idwt!(
    x::CuArray,
    b::WTOrthogonalBasis{N, F},
    l = 1;
    wpt = false
) :: Nothing where {N, F <: AbstractFloat}
    w = similar(gpux)
    idwt!(w, x, b, l; wpt = wpt)
    nothing
end

dwt(x, b, l = 1; wpt = false) =
    let c = copy(x)
        dwt!(c, b, l, wpt = wpt)
        c
    end

idwt(x, b, l = 1; wpt = false) =
    let c = copy(x)
        idwt!(c, b, l, wpt = wpt)
        c
    end
