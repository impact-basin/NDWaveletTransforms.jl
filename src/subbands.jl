## subbands.jl -- helpers for working with subbands.
## For the most part, you will want @wtview and friends.
## This will allow you to write e.g.
##
## @wtview a[:ll] .= 0
##
## To set all scaling coefficients to zero.

function wt_index_1d(r::UnitRange, i::Char)
    @match i begin
        'l' || 'L' => r.start:(r.start+r.stop-1)>>1
        'h' || 'H' => (r.start+r.stop+1)>>1:r.stop
        _   => r
    end
end

function wt_index_1d(r::UnitRange, i::String)
    length(i) == 0 && return r
    @match i[1] begin
        'l' || 'L' =>
            wt_index_1d(r.start:(r.start+r.stop-1)>>1, i[2:end])
        'h' || 'H' =>
            wt_index_1d((r.start+r.stop+1)>>1:r.stop,  i[2:end])
        _   => r
    end
end

function wt_index(sz::NTuple{N,UnitRange}, i::String) where N
    length(i) % N == 0 || @error "wt_index(): Bad index!" sz i
    Tuple(wt_index_1d(sz[k], i[k:N:end]) for k = 1:N)
end

wt_index(sz::NTuple{N,UnitRange}, i::Symbol) where N =
    wt_index(sz, string(i))

wt_index(sz::NTuple{N,UnitRange}, i::String...) where N =
    wt_index(sz, prod(i))

wt_index(sz::NTuple{N,UnitRange}, i::Symbol...) where N =
    wt_index(sz, prod(string.(i)))

wt_index(v::AbstractArray{T,N}, i::Union{String, Symbol}) where {T <: Number, N} =
    view(v, wt_index(Tuple(1:s for s in size(v)), i)...)

wt_index(v::AbstractArray{T,N}, i::Union{String, Symbol}...) where {T <: Number, N} =
        wt_index(v, prod(string(x) for x in i))

wt_index(i::Union{String, Symbol}...) =
    (v::AbstractArray{T,N} where {T<:Number, N}) -> 
        wt_index(v, prod(string(x) for x in i))

macro wtview(expr)
    # descend the AST
    return postwalk(expr) do x

        # look for expressions indexing into arrays
        @capture(x, arr_Symbol[syms__]) || return x

        # check that we have raw quotenodes
        all(x -> x isa QuoteNode, syms) || return x

        # check that those quotenodes resolve to symbols
        syms = prod(map(x -> string(x.value), syms))

        # invoke wtview with the list of symbols
        return :(wt_index($arr, $syms))
    end |> esc
end

sbviews(x :: T) where {N <: Number, T <: AbstractArray{N,1}} = (
    wt_index(x, :l),
    wt_index(x, :h),
)

sbviews(x :: T) where {N <: Number, T <: AbstractArray{N,2}} = (
    wt_index(x, :ll), wt_index(x, :lh),
    wt_index(x, :hl), wt_index(x, :hh),
)

subspaces(n::T) where T <: Number =
    n > 1 ? ['l' .* subspaces(n-1); 'h' .* subspaces(n-1)] :
    n > 0 ? ["l", "h"] :
            ["",  "" ]

subspaces(n::T, m::U) where {T<:Number, U<:Number} =
    (n <= 0 || m <= 0) ? [""] : [prod(x) for x in Iterators.product([subspaces(n) for i=1:m]...)][:]

function subspaces(
    w::A,
    x::A,
    packet = false
) :: Vector{Tuple} where {T <: AbstractFloat, N, A <: AbstractArray{T, N}}
    packet || return [(
        wt_index(w, repeat("l", N)),
        wt_index(x, repeat("l", N)),
    )]
    return [(
        wt_index(w, str),
        wt_index(x, str),
    ) for str in subspaces(N)]
end

