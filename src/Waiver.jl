module Waiver

using StyledStrings
using StaticArrays
using Memoization
using Primes
using LinearAlgebra
using Match
using Base.Threads
using PrecompileTools
using MacroTools: @capture, postwalk, prewalk
using MacroTools: splitdef, splitarg, rmlines
using MacroTools: prettify, unblock
using Base.Iterators
using FLoops
using EllipsisNotation
using Strided
# using Strided
using TiffImages

include("algorithms.jl")
export complement
export mprime
export cyclespin!
export cyclespinning!
export @turbofun

include("bases.jl")
export WT_LEVELS
export WAVELET_F
export SIGNAL_1D
export SIGNAL_2D
export WTBasis
export WT_LEVELS
export WTOrthogonalBasis
export WT_HAAR 
export WT_D1,      WT_D2,      WT_D3,    WT_D4,    WT_D5
export WT_D6,      WT_D7,      WT_D8,    WT_D9,    WT_D10
export WT_D11,     WT_D12,     WT_D13,   WT_D14,   WT_D15
export WT_D16,     WT_D17,     WT_D18,   WT_D19,   WT_D20
export WT_COIF2,   WT_COIF4,   WT_COIF6, WT_COIF8, WT_COIF10 
export WT_SYM4,    WT_SYM5,    WT_SYM6,  WT_SYM7,  WT_SYM8
export WT_SYM9,    WT_SYM10
export WT_BATTLE2, WT_BATTLE4, WT_BATTLE6
export WT_BEYL 
export WT_VAID 
# export WTBandIndex


include("utils.jl")
export nextk2n
export prevk2n
export pad
export upsample2
export wtlevels
export testimage
export mtprime
export mapbits
export bands
export wt_index_1d
export wt_index
export wtview
export @wtview
export wpt_bands
export sbviews
export subspaces

# include("gpu-kernels.jl")
# include("gpu-dispatch.jl")
# export dwt!
# export idwt!
# export dwt
# export idwt

include("innerloops.jl")
export _dwt_inner_loop!
export _idwt_inner_loop!
export _dwt!
export _idwt!

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

include("precompile.jl")

end # module Waiver
