# subbands.jl -- helpers for working with subbands.
# For the most part, you will want @rtview, which lets you write
#
#     @rtview a[:ll] .= 0
#
# to set all scaling coefficients to zero.

"""
    rtree_views(x)

Return every first-level subband of `x` as a tuple of views, in the order
used by [`rtree_view`](@ref). Band `1` is the approximation band and band
`2^ndims(x)` is high along every axis.

Each view is half the length of `x` along every axis, so on a multi-level
coefficient array `rtree_views` returns the bands of the finest level.
"""
@generated function rtree_views(x::T) where {E, N, T <: AbstractArray{E, N}}
    inds = [((i & (1<<(N-j))) == 0 ?
                :(1:size(x, $j)>>1) :
                :(size(x, $j)>>1 + 1:size(x, $j))
                for j in 1:N) for i=0:2^N - 1]
    exprs = [:(view(x, $(inds[i]...))) for i=1:2^N]
    return quote
        ($(exprs...),)
    end
end

"""
    rtree_view(x, band)

Return a view of one subband of `x`, the coefficient array of a wavelet
transform.

`band` is an `Int` index into [`rtree_views`](@ref), or a `Symbol` or
`String` naming the band. Names read one letter per axis, `l` for low and
`h` for high, so `:ll` is the 2-D approximation band, `:lh` is low along
axis 1 and high along axis 2, and `:hh` is high along both. Indices run
from `1` for all-low to `2^ndims(x)` for all-high.

A name must have one letter per axis. `_` leaves that axis whole, so `:l_`
is the low band along axis 1 across every axis-2 coefficient, and `:_l` is
the low band along axis 2 across every axis-1 coefficient. This reaches the
bands of an asymmetric transform: after `dwt!(x, WT_HAAR, (1, 2))`,
`@rtview x[:ll, :_l]` is the approximation band one level deep along axis 1
and two along axis 2.

Chaining navigates a multi-level transform. After `dwt!(x, WT_D4, 2)`,
`rtree_view(rtree_view(x, :ll), :ll)`, or `@rtview x[:ll, :ll]`, is the
coarsest approximation band. The result is a `SubArray`, so writing to it
writes to `x`.
"""
Base.@constprop :aggressive rtree_view(x, i::Int) = rtree_views(x)[i]

@inline function band_axis(n::Int, c::Char)
    if c == 'l' || c == 'L' # approximation band
        return 1:(n >> 1)
    elseif c == 'h' || c == 'H' # detail band
        return (n >> 1) + 1:n
    elseif c == '_' # both bands
        return 1:n 
    end
    throw(ArgumentError("invalid subband character $(repr(c))"))
end

Base.@constprop :aggressive function band_ranges(x, s::String)
    N = ndims(x)
    length(s) == N || throw(ArgumentError(
        "subband name \"$s\" has $(length(s)) letters; array has $N dimensions"))
    return ntuple(i -> band_axis(size(x, i), s[i]), Val(N))
end

Base.@constprop :aggressive rtree_view(x, s::String) =
    view(x, band_ranges(x, s)...)

Base.@constprop :aggressive rtree_view(x, s::Symbol) =
    rtree_view(x, String(s))


Base.@constprop :aggressive rtree_view(x, s::Vector{Symbol}) = length(s) == 1 ?
    rtree_view(x, s[1]) :
    rtree_view(rtree_view(x, s[1:end-1]), s[end])

Base.@constprop :aggressive rtree_view(x, s...) =
    rtree_view(x, collect(s))

const rtview  = rtree_view
const rtviews = rtree_views

isbandname(ind) = ind isa Symbol ||
                  (ind isa QuoteNode && ind.value isa Symbol) ||
                  ind isa AbstractString

"""
    @rtview x[band]

Rewrite subband indexing into calls to [`rtree_view`](@ref). The result is
a `SubArray`, so the macro works on either side of an assignment or a
broadcast:

    @rtview a[:ll] .= 0       # zero the approximation band
    b = @rtview a[:hl, :hh]   # HH band of the HL band
    @rtview a[:ll, :_l]       # axis 2 twice down, axis 1 once

Multiple bands chain from the outside in, so `_` keeps whatever range an
earlier name established. Only references whose indices are all band names
(`Symbol`, `String`, or `QuoteNode`) are rewritten; `a[1]` and `a[1:4]` are
left alone.
"""
macro rtview(expr)
    rv  = GlobalRef(@__MODULE__, :rtree_view)
    out = postwalk(expr) do e
        @capture(e, x_[inds__]) || return e
        all(isbandname, inds) || return e
        e = :($rv($x, $(inds[1])))
        for ind in inds[2:end]
            e = :($rv($e, $ind))
        end
        return e
    end
    return esc(out)
end

subspaces(w, x, wpt) = wpt ?
    zip(rtree_views(w), rtree_views(x)) :
    ((rtree_view(w, 1), rtree_view(x, 1)),)

"`rtview()`: Alias for `rtree_view()`."
const rtview  = rtree_view
"`rtviews()`: Alias for `rtree_views()`."
const rtviews = rtree_views
