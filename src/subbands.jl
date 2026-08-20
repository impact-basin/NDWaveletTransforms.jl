## subbands.jl -- helpers for working with subbands.
## For the most part, you will want @wtview and friends.
## This will allow you to write e.g.
##
## @wtview a[:ll] .= 0
##
## To set all scaling coefficients to zero.

@generated function rtree_views(x::T) :: Tuple{SubArray{E, N, T}} where {E, N, T <: AbstractArray{E, N}}  
    inds = [((i & (1<<(N-j))) == 0 ?
                :(1:size(x, $j)>>1) :
                :(size(x, $j)>>1 + 1:size(x, $j))
                for j in 1:N) for i=0:2^N - 1]
    exprs = [:(view(x, $(inds[i]...))) for i=1:2^N]
    return quote
        ($(exprs...),)
    end
end

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

macro rtview(expr)
    postwalk(expr) do e
        @capture(e, x_[inds__]) || return e
        e = :(rtree_view($(esc(x)), $(inds[1])))
        for ind in inds[2:end]
            e = :(rtree_view($e, $ind))
        end
        return e
    end
end

macro wtview(expr)
    return quote
        @rtview $expr
    end
end

subspaces(w, x, wpt) = wpt ?
    (rtree_views(w),    rtree_views(x)) :
    ((rtree_view(w, 1), rtree_view(x, 1)),)
