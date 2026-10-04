# processing.jl -- coefficient processing: shrinkage, denoising, compression.

# ---------------------------------------------------------------------------
# shrinkage
# ---------------------------------------------------------------------------

"""
    threshold!(x, λ; mode = :soft)

Shrink every entry of `x` toward zero in place, and return `x`. `x` is
usually a coefficient array from [`dwt`](@ref). `mode` selects the rule:

- `:soft` (the default) subtracts `λ` from the magnitude,
  `sign(x) * max(|x| - λ, 0)`;
- `:hard` keeps `x` where `|x| > λ` and zeroes it otherwise;
- `:garrote` applies `x * (1 - λ^2 / |x|^2)` where `|x| > λ`.

# Examples

```julia
c = dwt(s, WT_D4, 4)
threshold!(c, 0.1; mode = :soft)
s_denoised = idwt!(c, WT_D4, 4)
```
"""
function threshold!(x::AbstractArray{T}, λ; mode::Symbol = :soft) where {T <: Number}
    λ = abs(λ)
    if mode === :hard
        @inbounds for i in eachindex(x)
            x[i] = abs(x[i]) > λ ? x[i] : zero(T)
        end
    elseif mode === :soft
        @inbounds for i in eachindex(x)
            a = abs(x[i])
            x[i] = a > λ ? (x[i] / a) * (a - λ) : zero(T)
        end
    elseif mode === :garrote
        @inbounds for i in eachindex(x)
            a2 = abs2(x[i])
            x[i] = a2 > λ^2 ? x[i] * (1 - λ^2 / a2) : zero(T)
        end
    else
        throw(ArgumentError("mode must be :hard, :soft or :garrote, got $mode"))
    end
    return x
end

# ---------------------------------------------------------------------------
# largest coefficients
# ---------------------------------------------------------------------------

"""
    keeplargest!(x, k)

Zero all but the `k` largest-magnitude entries of `x`, in place, and return
`x`. Ties at the cut are broken by position, so exactly `k` entries are
retained.

# Examples

```julia
c = dwt(img, WT_D4, 3)
keeplargest!(c, round(Int, 0.05 * length(c)))
compressed = idwt!(c, WT_D4, 3)
```
"""
function keeplargest!(x::AbstractArray, k::Integer)
    n = length(x)
    k = clamp(Int(k), 0, n)
    k == n && return x
    if k == 0
        fill!(x, zero(eltype(x)))
        return x
    end
    mags = vec(abs.(x))
    cut = partialsort(mags, k; rev = true)
    keep = k - count(>(cut), mags)
    seen = 0
    @inbounds for i in eachindex(mags)
        a = mags[i]
        if a < cut
            x[i] = zero(eltype(x))
        elseif a == cut
            seen += 1
            seen > keep && (x[i] = zero(eltype(x)))
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# noise estimate
# ---------------------------------------------------------------------------

# median(|N(0, σ)|) = 0.6745 σ
const _MAD = 0.6744897501960817

"""
    noisiness(x)

Estimate the standard deviation of additive Gaussian noise from an array of
detail coefficients, as `median(abs, x) / 0.6745`.

Pass the finest-scale detail coefficients, which carry almost no signal.
[`denoise`](@ref) picks that band for you.
"""
noisiness(x::AbstractArray) = median(abs, x) / _MAD

# ---------------------------------------------------------------------------
# sparsity
# ---------------------------------------------------------------------------

"""
    sparsity(x)

Hoyer's sparsity of `x`, between `0` for a uniform array and `1` for a single
nonzero entry. Apply it to a coefficient array to compare how well two bases
concentrate the same signal.
"""
function sparsity(x::AbstractArray{T}) where {T <: Number}
    n = length(x)
    n <= 1 && return 0.0
    l1 = sum(abs, x)
    l2 = sqrt(float(sum(abs2, x)))
    (l1 == 0 || l2 == 0) && return 0.0
    return (sqrt(n) - l1 / l2) / (sqrt(n) - 1)
end

# ---------------------------------------------------------------------------
# subband energy
# ---------------------------------------------------------------------------

# name of the i-th first-level band of an N-dimensional array
function _rtree_name(i::Int, N::Int)
    chars = [((i - 1) & (1 << (N - j))) == 0 ? 'l' : 'h' for j in 1:N]
    return Symbol(String(chars))
end

"""
    rtenergy(c)

Energy `sum(abs2, ...)` of every first-level subband of the coefficient array
`c`, as a `NamedTuple` keyed by band name: `:ll`, `:lh`, `:hl`, `:hh` in two
dimensions, `:l` and `:h` in one.

The bands are those of [`rtree_views`](@ref), so recursing into `:ll` gives
the next level.

# Examples

```julia
c = dwt(img, WT_D4, 2)
e = rtenergy(c)
e.ll / sum(abs2, c)     # share of the energy in the approximation band
```
"""
function rtenergy(c::AbstractArray)
    N = ndims(c)
    names = ntuple(i -> _rtree_name(i, N), 2^N)
    energies = map(v -> sum(abs2, v), rtree_views(c))
    return NamedTuple{names}(energies)
end

# ---------------------------------------------------------------------------
# threshold rules
# ---------------------------------------------------------------------------

function _sure_threshold(c, σ)
    a = sort!(vec(abs.(c)))
    n = length(a)
    σ2 = float(σ)^2
    cum = 0.0
    best = Inf
    bestt = 0.0
    @inbounds for i in 1:n
        cum += a[i]^2
        risk = σ2 * n - 2 * σ2 * i + cum + (n - i) * a[i]^2
        if risk < best
            best = risk
            bestt = a[i]
        end
    end
    return bestt
end

function _threshold_value(rule, c, σ)
    if rule === :universal
        return σ * sqrt(2 * log(length(c)))
    elseif rule === :sure
        return _sure_threshold(c, σ)
    elseif rule isa Real
        return float(rule)
    else
        throw(ArgumentError("rule must be :universal, :sure or a number, got $rule"))
    end
end

# the finest-scale details: everything outside the level-1 approximation band
function _finest_details(c)
    mask = trues(size(c))
    rtree_view(mask, 1) .= false
    return c[mask]
end

# ---------------------------------------------------------------------------
# denoising
# ---------------------------------------------------------------------------

"""
    denoise(x, b, l; rule = :universal, mode = :soft, sigma = nothing, cycles = 0)

Denoise `x` by thresholding its wavelet coefficients and inverting.

`rule` selects the threshold: `:universal` is the VisuShrink threshold
`σ sqrt(2 log n)`, `:sure` is Stein's unbiased risk estimate, and a number is
used as the threshold directly. The noise `σ` is estimated from the
finest-scale detail coefficients with [`noisiness`](@ref) unless the `sigma`
keyword supplies it. `mode` is passed to [`threshold!`](@ref).

`cycles > 0` runs the whole denoiser under that many circular shifts with
[`cyclespinning!`](@ref), which suppresses boundary artefacts.

# Examples

```julia
s_denoised = denoise(s, WT_D4, 4)
s_smooth   = denoise(s, WT_D4, 4; cycles = 8)
```
"""
function denoise(x::AbstractArray, b, l; rule = :universal, mode::Symbol = :soft,
                 sigma = nothing, cycles::Integer = 0)
    work = v -> _denoise!(v, b, l, rule, mode, sigma)
    if cycles > 0
        y = copy(x)
        cyclespinning!(work, y, cycles)
        return y
    end
    return work(copy(x))
end

# the detail coefficients: everything outside the coarsest approximation band
function _detail_mask(c, l)
    N = ndims(c)
    ls = l isa Integer ? ntuple(_ -> Int(l), N) : Tuple(Int.(l))
    ranges = ntuple(d -> 1:(size(c, d) >> ls[d]), N)
    mask = trues(size(c))
    view(mask, ranges...) .= false
    return mask
end

function _denoise!(y, b, l, rule, mode, sigma)
    c = dwt(y, b, l)
    mask = _detail_mask(c, l)
    d = c[mask]
    σ = sigma === nothing ? noisiness(_finest_details(c)) : sigma
    threshold!(d, _threshold_value(rule, d, σ); mode = mode)
    c[mask] .= d
    y .= idwt!(c, b, l)
    return y
end

# ---------------------------------------------------------------------------
# compression
# ---------------------------------------------------------------------------

"""
    compress(x, b, l; keep = 0.05)

Compress `x` by keeping the largest `keep` fraction of its wavelet
coefficients and inverting. Returns a `NamedTuple` with the reconstruction
`x`, the number of coefficients `kept`, and the fraction of the energy
`energy` retained.

# Examples

```julia
result = compress(img, WT_D4, 3; keep = 0.05)
result.x        # the reconstruction
result.energy   # fraction of the energy kept
```
"""
function compress(x::AbstractArray, b, l; keep::Real = 0.05)
    c = dwt(x, b, l)
    total = sum(abs2, c)
    k = clamp(round(Int, keep * length(c)), 0, length(c))
    keeplargest!(c, k)
    x̂ = idwt!(c, b, l)
    energy = total == 0 ? 1.0 : sum(abs2, x̂) / total
    return (x = x̂, kept = k, energy = energy)
end

