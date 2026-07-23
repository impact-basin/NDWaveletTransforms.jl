using KernelAbstractions.Extras.LoopInfo: @unroll

@kernel inbounds=true function _dwt_1d_kernel!(
    sb,
    @Const(x),
    @Const(φ :: SVector{N, T}),
) where {N, T}
    j = @index(Global)
    n = length(x)

    offset = 2 * (j - 1)
    acc = zero(T)

    @unroll for k in 1:N
        idx = mod1(offset + k, n)
        acc += φ[k] * x[idx]
    end

    sb[j] = acc
end

@kernel inbounds=true function _idwt_1d_kernel!(
    w,
    @Const(ls),
    @Const(hs),
    @Const(φ :: SVector{N, T}),
    @Const(ψ :: SVector{N, T}),
) where {N, T}
    i = @index(Global)
    l = length(ls)
    acc = zero(T)

    start_k = 2 - (i & 1)
    @unroll for k in start_k:2:N
        j = mod1((i - k) ÷ 2 + 1, l)
        acc += φ[k] * ls[j]
        acc += ψ[k] * hs[j]
    end

    w[i] = acc
end

@kernel inbounds=true function _dwt_2d_kernel!(
    s, @Const(x), @Const(φ::SMatrix{N, N, T})
) where {N, T}
    i_out, j_out = @index(Global, NTuple)

    Nx   = size(x, 1)
    Ny   = size(x, 2)

    offset_i = 2 * (i_out - 1)
    offset_j = 2 * (j_out - 1)

    acc = zero(T)

    @unroll for k in 1:N
        i_in = mod1(offset_i + k, Nx)
        @unroll for m in 1:N
            j_in = mod1(offset_j + m, Ny)
            acc += φ[k, m] * x[i_in, j_in]
        end
    end

    s[i_out, j_out] = acc
end


@kernel inbounds=true function _idwt_2d_kernel!(
    w,
    @Const(ll), @Const(lh),
    @Const(hl), @Const(hh),
    @Const(φφ::SMatrix{K, K, T}), @Const(ψφ::SMatrix{K, K, T}),
    @Const(φψ::SMatrix{K, K, T}), @Const(ψψ::SMatrix{K, K, T}),
) where {K, T}
    i, j = @index(Global, NTuple)

    Nx    = size(w, 1)
    Ny    = size(w, 2)
    N2x   = Nx ÷ 2
    N2y   = Ny ÷ 2

    acc = zero(T)

    start_k = 2 - (i & 1)
    start_m = 2 - (j & 1)

    for k in start_k:2:K
        p = mod1((i - k) ÷ 2 + 1, N2x)
        for m in start_m:2:K
            q = mod1((j - m) ÷ 2 + 1, N2y)

            acc += ll[p, q] * φφ[k, m]
            acc += lh[p, q] * φψ[k, m]
            acc += hl[p, q] * ψφ[k, m]
            acc += hh[p, q] * ψψ[k, m]
        end
    end

    w[i, j] = acc
end

@kernel inbounds=true function _dwt_row_kernel!(
    out_l, out_h, @Const(x),
    @Const(φ::SVector{N, T}),
    @Const(ψ::SVector{N, T}),
) where {N, T}
    i_out, j = @index(Global, NTuple)

    Nx    = size(x, 1)

    offset_i = 2 * (i_out - 1)

    acc_l = zero(T)
    acc_h = zero(T)

    @unroll for k in 1:N
        i_in = mod1(offset_i + k, Nx)
        val  = x[i_in, j]
        acc_l += φ[k] * val
        acc_h += ψ[k] * val
    end

    out_l[i_out, j] = acc_l
    out_h[i_out, j] = acc_h
end

@kernel inbounds=true function _dwt_col_kernel!(
    out_l, out_h, @Const(x),
    @Const(φ::SVector{N, T}),
    @Const(ψ::SVector{N, T}),
) where {N, T}
    i, j_out = @index(Global, NTuple)

    Ny    = size(x, 2)

    offset_j = 2 * (j_out - 1)

    acc_l = zero(T)
    acc_h = zero(T)

    @unroll for k in 1:N
        j_in = mod1(offset_j + k, Ny)
        val  = x[i, j_in]
        acc_l += φ[k] * val
        acc_h += ψ[k] * val
    end

    out_l[i, j_out] = acc_l
    out_h[i, j_out] = acc_h
end

@kernel inbounds=true function _idwt_row_kernel!(
    out, @Const(low), @Const(high),
    @Const(φ::SVector{N, T}),
    @Const(ψ::SVector{N, T}),
) where {N, T}
    i, j = @index(Global, NTuple)

    N2x   = size(low, 1)

    start_k = 2 - (i & 1)
    acc = zero(T)

    @unroll for k in start_k:2:N
        p     = mod1((i - k) ÷ 2 + 1, N2x)
        acc  += low[p, j] * φ[k] + high[p, j] * ψ[k]
    end

    out[i, j] = acc
end

@kernel inbounds=true function _idwt_col_kernel!(
    out, @Const(low), @Const(high),
    @Const(φ::SVector{N, T}),
    @Const(ψ::SVector{N, T}),
) where {N, T}
    i, j = @index(Global, NTuple)

    N2y   = size(low, 2)

    start_k = 2 - (j & 1)
    acc = zero(T)

    @unroll for k in start_k:2:N
        p     = mod1((j - k) ÷ 2 + 1, N2y)
        acc  += low[i, p] * φ[k] + high[i, p] * ψ[k]
    end

    out[i, j] = acc
end
