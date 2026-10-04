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

Chaining navigates a multi-level transform. After `dwt!(x, WT_D4, 2)`,
`rtree_view(rtree_view(x, :ll), :ll)`, or `@rtview x[:ll, :ll]`, is the
coarsest approximation band. The result is a `SubArray`, so writing to it
writes to `x`.
"""
Base.@constprop :aggressive rtree_view(x, i::Int) = rtree_views(x)[i]

Base.@constprop :aggressive lh_str_to_num(s :: String) =
    parse(Int,
        replace(s, r"(L|l)" => s"0", r"(H|h)" => s"1");
        base=2
    ) + 1

Base.@constprop :aggressive rtree_view(x, s::String) =
    rtree_view(x, lh_str_to_num(s))

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

Multiple bands chain from the outside in. Only references whose indices are
all band names (`Symbol`, `String`, or `QuoteNode`) are rewritten; `a[1]` and
`a[1:4]` are left alone.
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
