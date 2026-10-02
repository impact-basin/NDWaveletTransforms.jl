import Wavelets as W
using NDWaveletTransforms
using Test
using Random

Random.seed!(20240901)

N = 1024

w = W.wavelet(W.WT.db2)
x = rand(N, N)

# Wavelets.jl applies the wavelet filter with a one-sample phase offset
# relative to the scaling filter (its detail coefficients are the aligned
# ones cyclically shifted by (N-2)/2 = 1 position). NDWaveletTransforms'
# `convention = :wavelets` reproduces Wavelets.jl's coefficient layout
# exactly; the default `:aligned` convention is the textbook phase.
@time w1 = W.dwt(x, w, 2)
@time w2 = dwt(x, WT_D2, 2; convention = :wavelets)

@test w1 ≈ w2

# The default convention is unchanged and still inverts exactly.
y = dwt(x, WT_D2, 2)
@test idwt(y, WT_D2, 2) ≈ x
@test dwt(x, WT_D2, 2) ≈ dwt(x, WT_D2, 2; convention = :aligned)

# Agreement with Wavelets.jl across bases, levels and dimensions.
for (bnd, wwt) in ((WT_HAAR, W.WT.haar), (WT_D2, W.WT.db2), (WT_D4, W.WT.db4))
    wf = W.wavelet(wwt)
    for l in (1, 2, 3)
        x1 = rand(64)
        @test dwt(x1, bnd, l; convention = :wavelets) ≈ W.dwt(x1, wf, l)
        x2 = rand(32, 64)
        @test dwt(x2, bnd, l; convention = :wavelets) ≈ W.dwt(x2, wf, l)
        # wavelet packets agree in :wavelets mode too
        @test wpt(x1, bnd, l; convention = :wavelets) ≈ W.wpt(x1, wf, l)
    end
end
