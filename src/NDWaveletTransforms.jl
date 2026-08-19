module NDWaveletTransforms

macro fastfun(expr)
    return quote
        Base.@constprop :aggressive Base.@propagate_inbounds $expr
    end |> esc
end

using StyledStrings
using StaticArrays
using Primes
using LinearAlgebra
using Match
using Base.Threads
using MacroTools: @capture, postwalk, prewalk
using MacroTools: splitdef, splitarg, rmlines
using MacroTools: prettify, unblock
using Base.Iterators
using FLoops
using EllipsisNotation
using Strided

include("algorithms.jl")
export complement
export cyclespin!
export cyclespinning!

include("bases.jl")
export WTBasis
export WTOrthogonalBasis
export WT_HAAR
export WT_D1,      WT_D2,      WT_D3,    WT_D4,    WT_D5,
       WT_D6,      WT_D7,      WT_D8,    WT_D9,    WT_D10,
       WT_D11,     WT_D12,     WT_D13,   WT_D14,   WT_D15,
       WT_D16,     WT_D17,     WT_D18,   WT_D19,   WT_D20
export WT_SYM2,    WT_SYM3,    WT_SYM4,  WT_SYM5,  WT_SYM6,
       WT_SYM7,    WT_SYM8,    WT_SYM9,  WT_SYM10
export WT_MB42,    WT_MB82,    WT_MB83,  WT_MB84,  WT_MB103,
       WT_MB123,   WT_MB143,   WT_MB163, WT_MB183, WT_MB243, WT_MB323
export WT_HAN23,   WT_HAN33,   WT_HAN45, WT_HAN55
export WT_COIF2,   WT_COIF4,   WT_COIF6, WT_COIF8, WT_COIF10
export WT_FK4,     WT_FK6,     WT_FK8,   WT_FK14,  WT_FK18,  WT_FK22
export WT_BATTLE2, WT_BATTLE4, WT_BATTLE6
export WT_BL7,     WT_BL9,     WT_BL10
export WT_BEYL
export WT_VAID

include("subbands.jl")
export wt_index_1d
export wt_index
export @wtview
export sbviews
export subspaces

include("innerloops.jl")
include("dwt.jl")
export dwt!
export dwt
export idwt!
export idwt
export wpt!
export wpt
export iwpt!
export iwpt

include("ns-dwt.jl")
export nsdwt!
export nsdwt
export nsidwt!
export nsidwt
export nswpt!
export nswpt
export nsiwpt!
export nsiwpt

end # module NDWaveletTransforms
