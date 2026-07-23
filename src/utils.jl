"""
    nextk2n(m, n)

    Find the smallest number of the form k 2^n (k, n ∈ 𝕫),
    such that k 2^n > m.
"""
nextk2n(m, n) = ((m >> n) + 1) << n

"""
    prevk2n(m, n)

    Find the largest number of the form k 2^n (k, n ∈ 𝕫),
    such that k 2^n  < m.
"""
prevk2n(m, n) = m - (m & ((1 << n) - 1))

"""
    pad(s :: Vector, n, L)

    Pad with a zero-order hold such that the minimum
    padding is given by the length of the filter n,
    and the resulting padded signal can be transformed
    at least L times, giving a pad length of
    nextk2n(length(s) + n, L).

    This padding is for 1D signals.
"""
function pad(s :: Vector{T},
             n :: Int,
             L :: Int) where T

    # compute minimum pad needed
    l = nextk2n(length(s) + n, L)
    sz = length(s)

    # pad the signal with a zero-order hold.
    sn = [s; Vector{T}(undef, l - sz)]
    sn[length(s):end] .= s[end]
    sn
end

"""
    pad(s :: Matrix, n, L :: WT_LEVELS)

    Pad with a zero-order hold such that the minimum
    padding in any direction is the length of the filter n,
    and the resulting padded signal can be transformed
    at least L times, giving a pad length of
    nextk2n(length(s) + n, L).

    This padding is for 2D signals. Importantly, this padding
    scheme differs from the 1D case in that the padding along
    rows and columns may be different due to different transform
    requirements.
"""
function pad(s :: Matrix{T}, n :: Int, L :: WT_LEVELS) where T

    # helper function to generate matrices
    m(x,y) = Matrix{T}(undef, x, y)

    # coerce L to a tuple
    L = @match L begin
        i :: Int => (i, i)
        _ => L
    end

    # compute padding size
    s1, s2 = size(s)
    p1 = nextk2n(s1 + 2n, L[1]) - s1
    p2 = nextk2n(s2 + 2n, L[2]) - s2

    @info "calculated sizes" s1 s2 p1 p2

    # make the new signal with block matrices
    r = [ m(p1, p2) m(p1, s2) m(p1, p2);
          m(s1, p2) s         m(s1, p2);
          m(p1, p2) m(p1, s2) m(p1, p2)
    ]

    # pad corners
    r[1:p1, 1:p2] .= s[1,1]
    r[1:p1, end-p2:end] .= s[1,end]
    r[end-p1:end, 1:p2] .= s[1,end]
    r[end-p1:end, end-p2:end] .= s[end,end]

    # pad top/bottom
    @threads for i=p1+1:p1+s1-1
        for j=1:p2
            r[i,j] = s[i-p1, 1]
        end
        for j=s2+p2:s2+2p2
            r[i,j] = s[i-p1, end]
        end
    end

    # pad left/right
    for j=p2+1:p2+s2-1
        for i=1:p1
            r[i,j] = s[1,j-p2]
        end
        for i=s1+p1:s1+2p1
            r[i,j] = s[end,j-p2]
        end
    end

    return r
end

function upsample2(x::AbstractArray{T}) where T
	ret = similar([x; x])
	ret[1:2:end] .= x[:]
	ret[2:2:end] .= T(0)
	ret
end

wtlevels(x::Int)             = trailing_zeros(x)
wtlevels(x::Tuple{Int, Int}) = (wtlevels(x[1]), wtlevels(x[2]))

function testimage(s::String) :: Matrix{Float64}
    dir = @__DIR__
    TiffImages.load(dir * "/testimages/" * s * ".tif") |> Matrix{Float64} |> rotr90
end


grayscale(x) = sqrt(0.299x.r^2 + 0.587x.g^2 + 0.114x.b^2)

minmax(x) = (minimum(x), maximum(x))


mapbits(f, x::UInt8)  = [f((x >> i) & 1) for i=0:7] |> reverse
to_binary(x, n=4) = bitstring(x)[end-n+1:end]
to_band(x, n=4) = replace(to_binary(x, n), '0' => "LL", "1" => "HL")
bands(n) = [to_band(i, n) for i=0:1<<n - 1]

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


function wtview(w::AbstractArray{T,N}, bands::Vector{Symbol}) where {T, N}
    idx = WTBandIndex(size(w))
    return view(w, idx(bands)[])
end

function wtview(w::AbstractArray{T, K}, bands::Vararg{Symbol, N}) where {T, K, N}
    return wtview(w, collect(bands))
end

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


wpt_bands(i, j) = @match (i, j) begin
    (k, l) where ((k >= 1 && l > 1) || (k > 1 && l >= 1)) =>
        let as = [:ll, :lh, :hl, :hh]
            bs = wpt_bands(k-1, l-1)
            [[a; b...] for a in as for b in bs]
        end
    (k, l) where (k > 1 && l == 0) =>
        let as = [:l_, :h_]
            bs = wpt_bands(k-1, 0)
            [[a; b...] for a in as for b in bs]
        end
    (k, l) where (k == 0 && l > 1) =>
        let as = [:_l, :_h]
            bs = wpt_bands(0, l-1)
            [[a; b...] for a in as for b in bs]
        end
    (k, l) where (k == 0 && l == 1) => [[:_l], [:_h]]
    (k, l) where (k == 1 && l == 0) => [[:l_], [:h_]]
    (k, l) where (k == 1 && l == 1) => [[:ll], [:lh], [:hl], [:hh]]
    (k, l) where (k <= 0 && l <= 0) => []
end
wpt_bands(i) = wpt_bands(i,i)

szinc(x) = let s = string(x)
    (s[1] in "hl" && s[2] in "hl") && return (2, 2)
     s[1] in "hl" && return (2, 1)
     s[2] in "hl" && return (1, 2)
     return (1,1)
end

_wtcascadesize(basis, indices) = @match indices begin
    [b, rest...] => szinc(b) .* _wtcascadesize(basis, @view(indices[2:end]))
    [] => (1,1)
end

sbviews(x :: T) where {N <: Number, T <: AbstractArray{N,1}} = (
    wt_index(x, :l),
    wt_index(x, :h),
)

sbviews(x) = (
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
