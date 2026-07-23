macro indexscheme(fn::Expr)
    @gensym f T v p i x y findtype U N
    d = splitdef(fn)
    !(length(d[:args]) ∈ [2, 3]) &&
        @error "bad function def" fn
    type = d[:name]
    d[:name] = f
    fn = combinedef(d)
    a = length(d[:args]) == 3 ?
        [i, :begin, :end]     :
        [i, :end]
    b = length(d[:args]) == 3 ?
        [x, :(firstindex($p, $y)), :(lastindex($p, $y))] :
        [x, :(lastindex($p, $y))]
    return quote
        @inline $fn
        struct $type{$U, $N, $T<:AbstractArray{$U, $N}} <: AbstractArray{$U, $N}
            $v :: $T
        end
        @inline Base.firstindex($p::$type{$U, $N, $T}) where {$U, $N, $T} = firstindex($p.$v)
        @inline Base.lastindex($p::$type{$U, $N, $T}) where {$U,$N,$T}         = lastindex($p.$v)
        @inline Base.getindex($p::$type{$U, $N, $T}, $i) where {$U,$N,$T}      = $p.$v[$f.($(a...))]
        @inline Base.setindex!($p::$type{$U, $N, $T}, $v, $i) where {$U,$N,$T} = $p.$v[$f.($(a...))] = $v
        @inline Base.getindex($p::$type{$U, $N, $T}, $i...) where {$U,$N,$T}   =
            $p.$v[($f.($(b...)) for ($y, $x) in enumerate($i))...]
        @inline Base.setindex!($p::$type{$U, $N, $T}, $v, $i...) where {$U,$N,$T} =
            $p.$v[($f.($(b...)) for ($y, $x) in enumerate($i))...] = $v
        @inline Base.getindex($p::$type{$U, $N, $T}, $i :: Base.CartesianIndex) where {$U,$N,$T}   =
            $p.$v[($f.($(b...)) for ($y, $x) in enumerate($i.I))...]
        @inline Base.setindex!($p::$type{$U, $N, $T}, $v, $i :: Base.CartesianIndex) where {$U,$N,$T} =
            $p.$v[($f.($(b...)) for ($y, $x) in enumerate($i.I))...] = $v
        @inline Base.view(p::$type{$U, $N, $T}, $i::$type) where {$U,$N,$T}    = $type(@view($p.$v[$f.($(a...))]))
        @inline Base.length($p::$type{$U, $N, $T}) where {$U,$N,$T}            = Base.length($p.$v)
        @inline Base.size($p::$type{$U, $N, $T}) where {$U,$N,$T}              = Base.size($p.$v)
        @inline Base.eltype($p::$type{$U, $N, $T}) where {$U,$N,$T}            = Base.eltype($p.$v)
        @inline Base.axes($p::$type{$U, $N, $T}) where {$U,$N,$T} = Base.axes($p.$v)
        @inline Base.axes($p::$type{$U, $N, $T}, $i) where {$U,$N,$T} = Base.axes($p.$v, $i)
        @inline Base.similar($p::$type{$U, $N, $T}) where {$U,$N,$T} = $type(Base.similar($p.$v))
        @inline Base.similar($p::$type{$U, $N, $T}, ::Type{TT}) where {$U,$N,$T, TT} = $type(TT.(Base.similar($p.$v)))
        @inline Base.Array($p::$type{$U, $N, $T}) where {$U,$N,$T} = $p.$v
        @inline Base.copy($p::$type{$U, $N, $T}) where {$U,$N,$T} = $type(copy($p.$v))
        @inline Base.iterate($p::$type{$U, $N, $T}) where {$U,$N,$T} = iterate($p.$v)
        @inline Base.iterate($p::$type{$U, $N, $T}, $i) where {$U,$N,$T} = iterate($p.$v, $i)
        @inline Base.IteratorSize($p::$type{$U, $N, $T}) where {$U,$N,$T} = Base.IteratorSize($p.$v)
        @inline Base.reshape($p::$type{$U, $N, $T}, $i) where {$U,$N,$T} = reshape($p.$v, $i)
        @inline Base.reshape($p::$type{$U, $N, $T}, $i...) where {$U,$N,$T} = reshape($p.$v, $i...)
        # broadcasting rules
        @inline Base.BroadcastStyle(::Type{$type{$U, $N, $T}}) where {$U,$N,$T} = Broadcast.ArrayStyle{$type{$U, $N, $T}}()
        @inline Base.broadcastable(p::$type{$U, $N, $T}) where {$U,$N,$T} = p
        @inline $findtype(bc::Base.Broadcast.Broadcasted) = $findtype(bc.args)
        @inline $findtype(args::Tuple) = $findtype($findtype(args[1]), Base.tail(args))
        @inline $findtype(x) = x
        @inline $findtype(::Tuple{}) = nothing
        @inline $findtype(x::$type{$U, $N, $T}, rest) where {$U,$N,$T} = x
        @inline $findtype(::Any, rest) = $findtype(rest)
        @inline Base.similar(
            bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{$type{$U, $N, $T}}},
            ::Type{TT},
        ) where {$U, $N, $T, TT} = similar($findtype(bc), TT)
    end |> esc
end

@indexscheme Periodic(x, e) = mod1(x, e)
@indexscheme ZOH(x, b, e)   = clamp(x, b, e)

x = hcat([1:10 |> collect for _ = 1:10]...)
# y = ZOH(@view(x[:, :]))
y = ZOH(x[1:end, 1:end])
y2 = sin.(y)


x = hcat([1:10 |> collect for _ = 1:10]...)
y = ZOH(@view(x[:, :]))
axes(y, 1)
axes(y, 2)
y[-15:20, -15:20] |> heatmap


x = 1:20 |> collect
y = ZOH(@view(x[:]))
y[0:25]

k = CartesianIndex(-1,3)
y[k]

x = hcat([1:10 |> collect for _ = 1:10]...)
y = Periodic(@view(x[:, :]))
axes(y, 1)
axes(y, 2)
y[-15:20, -15:20] |> heatmap


y[CartesianIndex(2,3)] = 5

heatmap()


# Base hold type.
"""
    Represents a padded signal using zero-order holds, etc.
"""
abstract type AbstractHold{T,N} <: AbstractArray{T,N} end

# AbstractHold interface functions, common to all holds.
Base.size(z::H)         where H <: AbstractHold = z.sz
Base.length(z::H)       where H <: AbstractHold = z.len
Base.IndexStyle(z::H)   where H <: AbstractHold = IndexStyle(z.s)
Base.firstindex(z::H)   where H <: AbstractHold = z.s[begin]
Base.lastindex(z::H)    where H <: AbstractHold = z.s[end]
Base.similar(z::H)      where H <: AbstractHold = H(similar(z.s), z.len, z.sz)
Base.axes(z::H)         where H <: AbstractHold = Tuple(1:s for s in z.sz) 
Base.copyto!(d::H,s::H) where H <: AbstractHold = d.s[:] .= s.s[:]

# Iteration interface rules
function Base.iterate(z::H)::Union{Nothing, Tuple} where H<:AbstractHold
    length(z) > 1 && (return z[1], 1)
    nothing
end

function Base.iterate(z::H, i::Int)::Union{Nothing,Tuple} where H<:AbstractHold
    i += 1
    i > length(z) ? nothing : (@inbounds z[i], i)
end

# broadcasting rules for holds
Base.BroadcastStyle(::Type{H}) where H <: AbstractHold = Broadcast.ArrayStyle{H}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{H}}, ::Type{T}) where {T, H<:AbstractHold}
    filter(a -> a isa H, bc.args)[1] |> similar
end

function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{H}}) where H <: AbstractHold
    h = filter(a -> a isa H, bc.args)[1]
    ret = similar(h) 
    @inbounds ret.s[:] .= h.s[:]
    ret
end

function Base.copyto!(dest::H, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{H}}) where H <: AbstractHold
    zoh = filter(a -> a isa H, bc.args)[1]
    @boundscheck size(dest) == size(zoh)
    @inbounds dest.s[:] = zoh.s[:]
end

## Hold type specifications

# a direct passthrough hold
struct NullHold{T, N, A<:AbstractArray{T,N}} <: AbstractHold{T,N}
    s :: A
    len :: Int
    sz :: NTuple{N, Int}

    function ZeroOrderHold(s::A) where {T,N,A<:AbstractArray{T,N}}
        new{T,N,A}(s, length(s), size(s))
    end

    function ZeroOrderHold(s::A, l::Int, sz::Tuple) where {T, N, A<:AbstractArray{T,N}}
        new{T,length(sz),A}(s, l, sz)
    end
end

Base.getindex(n::NullHold, i) = n.s[i]
Base.setindex!(n::NullHold, v, i) = n.s[i] = v

"""
    Represents a signal using a zero-order hold.
"""
struct ZeroOrderHold{T, N, A<:AbstractArray{T,N}} <: AbstractHold{T,N} 
    s   :: A
    len :: Int
    sz  :: NTuple{N,Int}

    function ZeroOrderHold(s::A) where {T,N,A<:AbstractArray{T,N}}
        new{T,N,A}(s, length(s), size(s))
    end

    #  function ZeroOrderHold(s::T) where T
        #  new{T,1,AbstractArray{T,1}}([s], 1, (1,))
    #  end

    function ZeroOrderHold(s::A, l::Int, sz::Tuple) where {T, N, A<:AbstractArray{T,N}}
        new{T,length(sz),A}(s, l, sz)
    end
end

# helper functions
clampindex(::Colon, s::T) where T <: Integer = 1:s
clampindex(index::T, s::T) where T <: Integer = clamp(index, 1:s)
clampindex(index :: Union{Vector, Tuple, AbstractRange}, s :: Integer) = clamp.(index, 1, s)
clampindex(index::CartesianIndex, s::Integer) = CartesianIndex(clamp.(index.I, 1, s)...)

function clampindex(index :: Union{Vector, Tuple},
                    s     :: Union{Vector, Tuple}) 
    [clampindex(i, d) for (i, d) in zip(index, s)]
end

Base.getindex(z::ZeroOrderHold, _::Tuple{Colon})            = @inbounds z.s[:]
Base.getindex(z::ZeroOrderHold, _::      Colon )            = @inbounds z.s[:]
Base.getindex(z::ZeroOrderHold, i::Tuple{CartesianIndex})   = z[i[1]]
function Base.getindex(z::ZeroOrderHold, i::Int) 
    i = clamp(i, 1, z.len) :: Int
    @inbounds z.s[i]
end
function Base.getindex(z::ZeroOrderHold, i::Tuple{T,T}) where T <: Integer
    @inbounds z.s[(clamp(i[1], 1, size(z.s[1])), clamp(i[2], 1, size(z.s[2])))...]
end
Base.getindex(z::ZeroOrderHold, i::CartesianIndex{1}) = @inbounds z.s[clamp(i.I[1], 1, length(z.s))]
Base.getindex(z::ZeroOrderHold, i::CartesianIndex{2}) = @inbounds z.s[clamp(i.I[1], 1, size(z.s)[1]), clamp(i.I[2], 1, size(z.s)[2])]
function Base.getindex(z::ZeroOrderHold, i...)
    # TODO: remove allocation from this function
    #  @infiltrate
    @inbounds z.s[clampindex(i, size(z))...]
end

isnullindex(index::Integer,
            s::Integer)              = index < 1 || index > s
isnullindex(index::CartesianIndex,
            s::Union{Vector, Tuple}) = any(index.I .< 1 .|| index.I .> s)

# get the OK indices as Boolean vectors
nonnullindex(i::Union{Vector,Tuple,AbstractRange},
             s::Union{Vector,Tuple,Integer}) = [i .> 1 .&& i .<= s]

function Base.setindex!(z::ZeroOrderHold, v, ::Union{Colon, Tuple{Colon}})
    z.s .= v
end

function Base.setindex!(z::ZeroOrderHold, v, a::Integer, b::Integer)
    s = size(z)
    isnullindex(a, s[1]) && return
    isnullindex(b, s[2]) && return
    z.s[a,b] = v
end

function Base.setindex!(z::ZeroOrderHold, v, i::Union{Vector,Tuple,AbstractRange})
    n = nonnullindex(i, size(z))
    length(n) > 0 || return
    @inbounds z.s[n] .= v[n]
end

function Base.setindex!(z::ZeroOrderHold, v, i::Integer)
    isnullindex(i, length(z)) && return
    @inbounds z.s[i] = v
end

function Base.setindex!(z::ZeroOrderHold, v, i::CartesianIndex)
    isnullindex(i, size(z)) && return
    @inbounds z.s[i] = v
end

Base.view(z::ZeroOrderHold, ::Colon) = ZeroOrderHold(view(z.s, Colon()))

function Base.view(z::ZeroOrderHold, args...)
    return ZeroOrderHold(view(z.s, clampindex(args, z.sz)...))
end

# hold with periodic bounds
struct PeriodicHold{T, N, A<:AbstractArray{T,N}} <: AbstractHold{T,N} 
    s   :: A
    len :: Int
    sz  :: NTuple{N,Int}

    function PeriodicHold(s::A) where {T,N,A<:AbstractArray{T,N}}
        new{T,N,A}(s, length(s), size(s))
    end

    function PeriodicHold(s::A, l::Int, sz::Tuple) where {T, N, A<:AbstractArray{T,N}}
        new{T,length(sz),A}(s, l, sz)
    end
end

# PeriodicHold getindex functions
Base.getindex(z::PeriodicHold, _::Tuple{Colon})            = @inbounds z.s[:]
Base.getindex(z::PeriodicHold, _::      Colon )            = @inbounds z.s[:]
Base.getindex(z::PeriodicHold, i::Tuple{CartesianIndex})   = z[i[1]]

function Base.getindex(z::PeriodicHold, i::Int) 
    i %= z.len
    i = i <= 0 ? z.len + i : i
    @inbounds z.s[i]
end

function Base.getindex(z::PeriodicHold, i::Tuple{T,T}) where T <: Integer
    j, k = i[1] % z.sz[1], i[2] % z.sz[2]
    j = j <= 0 ? z.sz[1] + j : j
    k = k <= 0 ? z.sz[1] + k : k
    @inbounds z.s[j, k]
end
Base.getindex(z::PeriodicHold, i::CartesianIndex{1}) = z[i.I[1]]
Base.getindex(z::PeriodicHold, i::CartesianIndex{2}) = z[i.I[1], i.I[2]]

# setindex functions for periodic holds. 
Base.setindex!(z::PeriodicHold, v, ::Union{Colon,Tuple{Colon}}) = z.s .= v

function Base.setindex!(z::PeriodicHold, v, a::Integer, b::Integer)
    a %= z.sz[1]
    b %= z.sz[2]
    a = a <= 0 ? z.sz[1] + a : a
    b = b <= 0 ? z.sz[2] + b : b
    z.s[a,b] = v
end

function Base.setindex!(z::PeriodicHold, v, i::Vector)
    i .%= z.sz
    i[i .<= 0] .+= z.sz[i .<= 0]
    @inbounds z.s[i] .= v[i]
end

function Base.setindex!(z::PeriodicHold, v, i::Tuple)
    n = Vector{Integer}(i .% z.sz)
    n[n .<= 0] .+= z.sz[n .<= 0]
    @inbounds z.s[n] .= v[n]
end

function Base.setindex!(z::PeriodicHold, v, i::CartesianIndex)
    z[i.I] = v
end

function Base.setindex!(z::PeriodicHold, v, i::Integer)
    i %= z.len
    i = i <= 0 ? z.len - i : i
    @inbounds z.s[i] = v
end

# views for periodic holds
Base.view(z::PeriodicHold, ::Colon) = PeriodicHold(view(z.s, Colon()))

function Base.view(z::PeriodicHold, args...)
    s = size(z)
    # TODO: fix this.
    return PeriodicHold(view(z.s))
end
