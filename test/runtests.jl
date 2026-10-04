using Test
using Random
using LinearAlgebra
using StaticArrays
using NDWaveletTransforms
import Wavelets as W
import FLoops
import Aqua

@testset "Correctness" begin
    @test dwt(ones(4, 4), WT_HAAR, 1) ≈ [
        2.0 2.0 0.0 0.0;
        2.0 2.0 0.0 0.0;
        0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0
    ]

    @test dwt(ones(4, 4), WT_HAAR, (1, 2)) ≈ [
        2.8284271247461894 0.0 0.0 0.0;
        2.8284271247461894 0.0 0.0 0.0;
        0.0000000000000000 0.0 0.0 0.0;
        0.0000000000000000 0.0 0.0 0.0
    ] 
end

@testset "Haar traforms" begin
    x = rand(Float32,128,128)
    y = dwt(x, WT_HAAR, 1)
    idwt!(y, WT_HAAR, 1)
    @test x ≈ y
    y = dwt(x, WT_HAAR, 2)
    idwt!(y, WT_HAAR, 2)
    @test x ≈ y
    y = dwt(x, WT_HAAR, (2, 4))
    idwt!(y, WT_HAAR, (2, 4))
    @test x ≈ y
end

@testset "Haar transforms, nonstandard" begin
    x = rand(Float32,128,128)
    y = nsdwt(x, WT_HAAR, 1)
    nsidwt!(y, WT_HAAR, 1)
    @test x ≈ y
    y = nsdwt(x, WT_HAAR, 2)
    nsidwt!(y, WT_HAAR, 2)
    @test x ≈ y
    y = nsdwt(x, WT_HAAR, (2, 4))
    nsidwt!(y, WT_HAAR, (2, 4))
    @test x ≈ y
end

@testset "D4" begin
    x = rand(Float32,128,128)
    y = dwt(x, WT_D4, 1)
    idwt!(y, WT_D4, 1)
    @test x ≈ y
    y = dwt(x, WT_D4, 2)
    idwt!(y, WT_D4, 2)
    @test x ≈ y
    y = dwt(x, WT_D4, (2, 4))
    idwt!(y, WT_D4, (2, 4))
    @test x ≈ y
end

@testset "D4, nonstandard" begin
    x = rand(Float32,128,128)
    y = nsdwt(x, WT_D4, 1)
    nsidwt!(y, WT_D4, 1)
    @test x ≈ y
    y = nsdwt(x, WT_D4, 2)
    nsidwt!(y, WT_D4, 2)
    @test x ≈ y
    y = nsdwt(x, WT_D4, (2, 4))
    nsidwt!(y, WT_D4, (2, 4))
    @test x ≈ y
end

@testset "Float64" begin
    x = rand(128,128)
    y = dwt(x, WT_HAAR, 1)
    idwt!(y, WT_HAAR, 1)
    @test x ≈ y
    y = dwt(x, WT_HAAR, 2)
    idwt!(y, WT_HAAR, 2)
    @test x ≈ y
    y = dwt(x, WT_HAAR, (2, 4))
    idwt!(y, WT_HAAR, (2, 4))
    @test x ≈ y
end

@testset "Float64, nonstandard" begin
    x = rand(128,128)
    y = nsdwt(x, WT_HAAR, 1)
    nsidwt!(y, WT_HAAR, 1)
    @test x ≈ y
    y = nsdwt(x, WT_HAAR, 2)
    nsidwt!(y, WT_HAAR, 2)
    @test x ≈ y
    y = nsdwt(x, WT_HAAR, (2, 4))
    nsidwt!(y, WT_HAAR, (2, 4))
    @test x ≈ y
end

@testset "Large transform" begin
    x = rand(256,1024)
    y = dwt(x, WT_HAAR, 1)
    idwt!(y, WT_HAAR, 1)
    @test x ≈ y
    y = dwt(x, WT_HAAR, 2)
    idwt!(y, WT_HAAR, 2)
    @test x ≈ y
    y = dwt(x, WT_HAAR, (2, 4))
    idwt!(y, WT_HAAR, (2, 4))
    @test x ≈ y
end

@testset "Large transform, nonstandard" begin
    x = rand(256,1024)
    y = nsdwt(x, WT_HAAR, 1)
    nsidwt!(y, WT_HAAR, 1)
    @test x ≈ y
    y = nsdwt(x, WT_HAAR, 2)
    nsidwt!(y, WT_HAAR, 2)
    @test x ≈ y
    y = nsdwt(x, WT_HAAR, (2, 4))
    nsidwt!(y, WT_HAAR, (2, 4))
    @test x ≈ y
end

@testset "Wavelet packet round trips, 1-D" begin
    for T in (Float32, Float64)
        for b in (WT_HAAR, WT_D4, WT_D8)
            for n in (4, 8, 16)
                x = rand(T, n)
                for l in (1, 2, 3)
                    y = wpt(x, b, l)
                    iwpt!(y, b, l)
                    @test x ≈ y
                end
            end
        end
    end
end

@testset "Wavelet packet round trips, 2-D" begin
    for T in (Float32, Float64)
        for b in (WT_HAAR, WT_D4)
            for (m, n) in ((8, 8), (16, 32))
                x = rand(T, m, n)
                for l in ((1, 1), (2, 2), (2, 3), (3, 2))
                    y = wpt(x, b, l)
                    iwpt!(y, b, l)
                    @test x ≈ y
                end
            end
        end
    end
end

@testset "Wavelet packet round trips, 3-D" begin
    for T in (Float32, Float64)
        x = rand(T, 8, 8, 8)
        for l in ((1, 1, 1), (2, 1, 2), (2, 2, 2), (3, 2, 1))
            y = wpt(x, WT_HAAR, l)
            iwpt!(y, WT_HAAR, l)
            @test x ≈ y
        end
    end
end

@testset "3-D transforms" begin
    for T in (Float32, Float64)
        x = rand(T, 8, 8, 8)
        for b in (WT_HAAR, WT_D4)
            for l in ((1, 1, 1), (2, 1, 2), (2, 2, 2), (3, 2, 1))
                y = dwt(x, b, l)
                idwt!(y, b, l)
                @test x ≈ y
                y = nsdwt(x, b, l)
                nsidwt!(y, b, l)
                @test x ≈ y
            end
        end
    end
end

@testset "Nonstandard wavelet packets" begin
    for T in (Float32, Float64)
        for (m, n) in ((8, 8), (16, 32))
            x = rand(T, m, n)
            for l in ((1, 1), (2, 2), (2, 3))
                y = nswpt(x, WT_D4, l)
                nsiwpt!(y, WT_D4, l)
                @test x ≈ y
            end
        end
    end
end

@testset "Nonstandard transforms, known values" begin
    sq2 = sqrt(2.0)
    # The nonstandard transform applies a level-l[axis] 1-D transform to
    # every line along each axis in turn; both axes must contribute.
    @test nsdwt(ones(4), WT_HAAR, 2) ≈ [2.0, 0.0, 0.0, 0.0]
    @test nsdwt(ones(8), WT_HAAR, 3) ≈ [2*sq2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lvl11 = [2.0 2.0 0.0 0.0; 2.0 2.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    @test nsdwt(ones(4, 4), WT_HAAR, (1, 1)) ≈ lvl11
    @test nsdwt(ones(4, 4), WT_HAAR, 1) ≈ lvl11
    # columns level-1 then rows level-2
    l12 = [2*sq2 0.0 0.0 0.0; 2*sq2 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    @test nsdwt(ones(4, 4), WT_HAAR, (1, 2)) ≈ l12
    # columns level-2 then rows level-1
    l21 = [2*sq2 2*sq2 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    @test nsdwt(ones(4, 4), WT_HAAR, (2, 1)) ≈ l21
    @test nswpt(ones(4, 4), WT_HAAR, (1, 2)) ≈ l12
end

@testset "Nonstandard transforms, forward correctness" begin
    # Reference: apply a level-l[axis] 1-D transform to every line along each
    # axis in turn, using the (separately verified) 1-D dwt!.
    function ref_nsdwt(x, b, level)
        N = ndims(x)
        n = size(x)
        y = copy(x)
        w = similar(y)
        for d in 1:N
            nlines = prod(n) ÷ n[d]
            for li in 0:nlines-1
                rem = li
                coords = ntuple(_ -> 1, N)
                for j in 1:N
                    if j != d
                        coords = Base.setindex(coords, mod(rem, n[j]) + 1, j)
                        rem = fld(rem, n[j])
                    end
                end
                ranges = ntuple(j -> j == d ? (1:n[d]) : coords[j], N)
                dwt!(view(y, ranges...), view(w, ranges...), b, [level[d]])
            end
        end
        y
    end
    for T in (Float32, Float64)
        for b in (WT_HAAR, WT_D4)
            for (m, n) in ((4, 4), (8, 8), (16, 32))
                x = rand(T, m, n)
                for l in ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2))
                    @test nsdwt(copy(x), b, l) ≈ ref_nsdwt(x, b, l)
                end
            end
            x = rand(T, 8, 8, 8)
            for l in ((1, 1, 1), (2, 1, 2), (1, 2, 3))
                @test nsdwt(copy(x), b, l) ≈ ref_nsdwt(x, b, l)
            end
        end
    end
end

@testset "Level vectors are not mutated" begin
    x = rand(8, 8)
    w = similar(x)
    for f in (dwt!, idwt!)
        l = [2, 3]
        lcopy = copy(l)
        f(copy(x), w, WT_D4, l)
        @test l == lcopy
    end
end

@testset "Known coefficient values (Haar)" begin
    sq2 = sqrt(2.0)
    @test wpt(ones(4), WT_HAAR, 1) ≈ [sq2, sq2, 0.0, 0.0]
    @test dwt(ones(8), WT_HAAR, 2) ≈ [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test wpt(ones(8), WT_HAAR, 2) ≈ [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test dwt(ones(8), WT_HAAR, 3) ≈ [2*sq2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test dwt(ones(8), WT_D4, 1) ≈ [sq2, sq2, sq2, sq2, 0.0, 0.0, 0.0, 0.0]
    lvl1 = [2.0 2.0 0.0 0.0; 2.0 2.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    @test wpt(ones(4, 4), WT_HAAR, 1) ≈ lvl1
    @test wpt(ones(4, 4), WT_HAAR, (1, 1)) ≈ lvl1
    @test nsdwt(ones(4, 4), WT_HAAR, 1) ≈ lvl1
    @test nswpt(ones(4, 4), WT_HAAR, 1) ≈ lvl1
    lvl2 = [4.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
    @test dwt(ones(4, 4), WT_HAAR, 2) ≈ lvl2
    @test wpt(ones(4, 4), WT_HAAR, (2, 2)) ≈ lvl2
end

@testset "Phase conventions" begin
    # Two phase conventions are provided via the `convention` keyword:
    #   :aligned  (default) -- both filters act on the same window (textbook)
    #   :wavelets          -- Wavelets.jl-compatible detail-coefficient phase
    # The scaling coefficients are identical in both; only the detail
    # coefficients differ, by a cyclic shift of (N-2)/2 positions.
    for T in (Float32, Float64)
        x = rand(T, 32, 32)
        @test dwt(x, WT_D4, 2) ≈ dwt(x, WT_D4, 2; convention = :aligned)
        # both conventions are orthogonal: round trips invert
        for conv in (:aligned, :wavelets)
            for (f, fi, args) in (
                (dwt, idwt, (WT_D4, 2)),
                (dwt, idwt, (WT_D4, (2, 3))),
                (wpt, iwpt, (WT_D4, 2)),
                (nsdwt, nsidwt, (WT_D4, (2, 1))),
                (nswpt, nsiwpt, (WT_D4, (1, 2))),
            )
                y = f(x, args...; convention = conv)
                @test x ≈ fi(y, args...; convention = conv)
            end
            x1 = rand(T, 32)
            y = dwt(x1, WT_D4, 2; convention = conv)
            @test x1 ≈ idwt(y, WT_D4, 2; convention = conv)
            y = wpt(x1, WT_D4, 2; convention = conv)
            @test x1 ≈ iwpt(y, WT_D4, 2; convention = conv)
        end
    end

    # The :wavelets convention reproduces Wavelets.jl's coefficient layout
    # exactly (this is the "same output" cross-check).
    for (bnd, wwt) in ((WT_HAAR, W.WT.haar), (WT_D2, W.WT.db2),
                       (WT_D3, W.WT.db3), (WT_D4, W.WT.db4))
        wf = W.wavelet(wwt)
        for l in (1, 2, 3)
            x = rand(64)
            @test dwt(x, bnd, l; convention = :wavelets) ≈ W.dwt(x, wf, l)
            @test wpt(x, bnd, l; convention = :wavelets) ≈ W.wpt(x, wf, l)
            x = rand(32, 64)
            @test dwt(x, bnd, l; convention = :wavelets) ≈ W.dwt(x, wf, l)
        end
    end

    # The default convention differs from Wavelets.jl for N > 2 taps (the
    # detail coefficients are cyclically shifted) but agrees for Haar.
    x = rand(64)
    wf = W.wavelet(W.WT.db2)
    @test dwt(x, WT_D2, 1) != W.dwt(x, wf, 1)
    @test dwt(x, WT_HAAR, 1) ≈ W.dwt(x, W.wavelet(W.WT.haar), 1)
end

@testset "rtree indexing: lengths OK" begin    x = rand(4)
    @test rtree_views(x) |> length == 2
    x = rand(4, 4)
    @test rtree_views(x) |> length == 4
    x = rand(4, 4, 4)
    @test rtree_views(x) |> length == 8
    x = rand(4, 4, 4, 4)
    @test rtree_views(x) |> length == 16
end

@testset "rtree: correct indexing" begin
    x = rand(4)
    @test rtree_view(x, 1) ≈ @view x[1:end>>1]
    @test rtree_view(x, 2) ≈ @view x[end>>1 + 1:end]
    x = rand(4, 4)
    @test rtree_view(x, 1) ≈ @view x[1:end>>1, 1:end>>1]
    @test rtree_view(x, 2) ≈ @view x[1:end>>1, end>>1 + 1:end]
    @test rtree_view(x, 3) ≈ @view x[end>>1 + 1:end, 1:end>>1]
    @test rtree_view(x, 4) ≈ @view x[end>>1 + 1:end, end>>1 + 1:end]
end

@testset "rtree: band index string to number" begin
    @test NDWaveletTransforms.lh_str_to_num("l") == 1
    @test NDWaveletTransforms.lh_str_to_num("h") == 2
    @test NDWaveletTransforms.lh_str_to_num("LL") == 1
    @test NDWaveletTransforms.lh_str_to_num("LH") == 2
    @test NDWaveletTransforms.lh_str_to_num("Hl") == 3
    @test NDWaveletTransforms.lh_str_to_num("hH") == 4
end

@testset "rtree: fancy band indexing" begin
    x = rand(4)
    @test rtree_view(x, 1) ≈ rtree_view(x, :l)
    @test rtree_view(x, 2) ≈ rtree_view(x, :h)
    x = rand(4, 4)
    @test rtree_view(x, 1) ≈ rtree_view(x, :ll)
    @test rtree_view(x, 2) ≈ rtree_view(x, :lh)
    @test rtree_view(x, 3) ≈ rtree_view(x, :hl)
    @test rtree_view(x, 4) ≈ rtree_view(x, :hh)
    x = rand(4, 4, 4)
    @test rtree_view(x, 1) ≈ rtree_view(x, :lll)
    @test rtree_view(x, 2) ≈ rtree_view(x, :llh)
    @test rtree_view(x, 3) ≈ rtree_view(x, :lhl)
    @test rtree_view(x, 4) ≈ rtree_view(x, :lhh)
    @test rtree_view(x, 5) ≈ rtree_view(x, :hll)
    @test rtree_view(x, 6) ≈ rtree_view(x, :hlh)
    @test rtree_view(x, 7) ≈ rtree_view(x, :hhl)
    @test rtree_view(x, 8) ≈ rtree_view(x, :hhh)
end

@testset "rtree: varargs/vectors" begin
    x = rand(4)
    @test rtree_view(x, :l) ≈ rtree_view(x, [:l])
    @test rtree_view(x, :h) ≈ rtree_view(x, [:h])
    x = rand(4, 4)
    rtree_view(x, :ll) ≈ rtree_view(x, [:ll])
    rtree_view(x, :lh) ≈ rtree_view(x, [:lh])
    rtree_view(x, :hl) ≈ rtree_view(x, [:hl])
    rtree_view(x, :hh) ≈ rtree_view(x, [:hh])
    x = rand(4, 4, 4)
    @test rtree_view(x, :lll) ≈ rtree_view(x, [:lll])
    @test rtree_view(x, :llh) ≈ rtree_view(x, [:llh])
    @test rtree_view(x, :lhl) ≈ rtree_view(x, [:lhl])
    @test rtree_view(x, :lhh) ≈ rtree_view(x, [:lhh])
    @test rtree_view(x, :hll) ≈ rtree_view(x, [:hll])
    @test rtree_view(x, :hlh) ≈ rtree_view(x, [:hlh])
    @test rtree_view(x, :hhl) ≈ rtree_view(x, [:hhl])
    @test rtree_view(x, :hhh) ≈ rtree_view(x, [:hhh])
    x = rand(8, 8)
    @test rtree_view(x, :ll, :ll) ≈ rtree_view(x, [:ll, :ll])
    @test rtree_view(x, :lh, :ll) ≈ rtree_view(x, [:lh, :ll])
    @test rtree_view(x, :hl, :ll) ≈ rtree_view(x, [:hl, :ll])
    @test rtree_view(x, :hh, :ll) ≈ rtree_view(x, [:hh, :ll])
    @test rtree_view(x, :ll, :ll) ≈ rtree_view(x, [:ll, :ll])
    @test rtree_view(x, :lh, :lh) ≈ rtree_view(x, [:lh, :lh])
    @test rtree_view(x, :hl, :hl) ≈ rtree_view(x, [:hl, :hl])
    @test rtree_view(x, :hh, :hh) ≈ rtree_view(x, [:hh, :hh])
end

@testset "@rtview macro, basic usage" begin
    x = rand(4)
    @test rtree_view(x, :l) ≈ @rtview x[:l]
    @test rtree_view(x, :h) ≈ @rtview x[:h]
    x = rand(4, 4)
    @test rtree_view(x, :ll) ≈ @rtview x[:ll]
    @test rtree_view(x, :lh) ≈ @rtview x[:lh]
    @test rtree_view(x, :hl) ≈ @rtview x[:hl]
    @test rtree_view(x, :hh) ≈ @rtview x[:hh]
    x = rand(4, 4, 4)
    @test rtree_view(x, :lll) ≈ @rtview x[:lll]
    @test rtree_view(x, :llh) ≈ @rtview x[:llh]
    @test rtree_view(x, :lhl) ≈ @rtview x[:lhl]
    @test rtree_view(x, :lhh) ≈ @rtview x[:lhh]
    @test rtree_view(x, :hll) ≈ @rtview x[:hll]
    @test rtree_view(x, :hlh) ≈ @rtview x[:hlh]
    @test rtree_view(x, :hhl) ≈ @rtview x[:hhl]
    @test rtree_view(x, :hhh) ≈ @rtview x[:hhh]
end

@testset "@rtview macro, multi-index" begin
    x = rand(8)
    @test rtree_view(rtree_view(x, :l), :l) ≈ @rtview x[:l, :l]
    @test rtree_view(rtree_view(x, :l), :h) ≈ @rtview x[:l, :h]
    @test rtree_view(rtree_view(x, :h), :l) ≈ @rtview x[:h, :l]
    @test rtree_view(rtree_view(x, :h), :h) ≈ @rtview x[:h, :h]
end

@testset "GPU (KernelAbstractions) transforms" begin
    gpu_ok = false
    try
        using CUDA
        gpu_ok = CUDA.functional()
    catch
        gpu_ok = false
    end
    if gpu_ok
        rng = MersenneTwister(7)
        # forward results must match the CPU implementation, and GPU round
        # trips must invert, for the standard, wavelet-packet and
        # nonstandard transforms.
        for T in (Float32, Float64)
            for n in (4, 8, 64)
                x = rand(rng, T, n)
                for b in (WT_HAAR, WT_D4)
                    for l in (1, 2, 3)
                        @test dwt(x, b, l) ≈ Array(dwt(CuArray(x), b, l))
                        y = wpt(CuArray(x), b, l); iwpt!(y, b, l)
                        @test x ≈ Array(y)
                    end
                end
            end
            for (m, n) in ((8, 8), (32, 32))
                x = rand(rng, T, m, n)
                for b in (WT_HAAR, WT_D4)
                    for l in ((1, 1), (2, 2), (2, 3), (3, 2))
                        @test dwt(x, b, l) ≈ Array(dwt(CuArray(x), b, l))
                        y = dwt(CuArray(x), b, l); idwt!(y, b, l)
                        @test x ≈ Array(y)
                        y = wpt(CuArray(x), b, l); iwpt!(y, b, l)
                        @test x ≈ Array(y)
                    end
                    for l in ((1, 1), (1, 2), (2, 1), (2, 3))
                        @test nsdwt(x, b, l) ≈ Array(nsdwt(CuArray(x), b, l))
                        y = nsdwt(CuArray(x), b, l); nsidwt!(y, b, l)
                        @test x ≈ Array(y)
                    end
                end
            end
            x = rand(rng, T, 8, 8, 8)
            for l in ((1, 1, 1), (2, 1, 2))
                @test dwt(x, WT_D4, l) ≈ Array(dwt(CuArray(x), WT_D4, l))
                y = dwt(CuArray(x), WT_D4, l); idwt!(y, WT_D4, l)
                @test x ≈ Array(y)
            end
        end
        # known values: both axes must contribute on the GPU too
        sq2 = sqrt(2.0)
        l12 = [2*sq2 0.0 0.0 0.0; 2*sq2 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        l21 = [2*sq2 2*sq2 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        @test Array(nsdwt(CuArray(ones(4, 4)), WT_HAAR, (1, 2))) ≈ l12
        @test Array(nsdwt(CuArray(ones(4, 4)), WT_HAAR, (2, 1))) ≈ l21
        # the :wavelets phase convention must match the CPU path and invert
        for T in (Float32, Float64)
            for n in (4, 8, 64)
                x = rand(rng, T, n)
                for b in (WT_HAAR, WT_D4)
                    for l in (1, 2, 3)
                        @test dwt(x, b, l; convention = :wavelets) ≈
                              Array(dwt(CuArray(x), b, l; convention = :wavelets))
                        y = dwt(CuArray(x), b, l; convention = :wavelets)
                        idwt!(y, b, l; convention = :wavelets)
                        @test x ≈ Array(y)
                    end
                end
            end
            for (m, n) in ((8, 8), (32, 32))
                x = rand(rng, T, m, n)
                for b in (WT_HAAR, WT_D4)
                    for l in ((1, 1), (2, 2), (2, 3))
                        @test dwt(x, b, l; convention = :wavelets) ≈
                              Array(dwt(CuArray(x), b, l; convention = :wavelets))
                        y = dwt(CuArray(x), b, l; convention = :wavelets)
                        idwt!(y, b, l; convention = :wavelets)
                        @test x ≈ Array(y)
                        @test nsdwt(x, b, l; convention = :wavelets) ≈
                              Array(nsdwt(CuArray(x), b, l; convention = :wavelets))
                    end
                end
            end
        end
    else
        @info "CUDA not available; skipping GPU tests"
    end
end

@testset "threaded passes are free of FLoops boxes" begin
    if Threads.nthreads() > 1
        # boxing is a warning by default; make it fail the suite
        FLoops.assistant(:error)
        try
            for conv in (:aligned, :wavelets)
                x = rand(128, 128, 32)
                y = copy(x)
                dwt!(x, WT_D8, 1; convention = conv)
                idwt!(x, WT_D8, 1; convention = conv)
                @test x ≈ y
            end
        finally
            FLoops.assistant(:warn)
        end
    else
        @info "single-threaded; skipping the FLoops box check"
    end
end

@testset "spin cycling" begin
    x = collect(1.0:8.0)
    y = copy(x)
    cyclespin!(y, 2)
    @test y == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0]

    z = copy(x)
    cyclespin!(z, 3)
    cyclespin!(z, -3)
    @test z == x

    w = rand(32, 32)
    before = copy(w)
    calls = Ref(0)
    cyclespinning!(w, 3) do v
        calls[] += 1
        v .+= 1
    end
    @test calls[] == 3
    @test w ≈ before .+ 1

    @test complement(SVector{2, Float64}(1, -1)) == SVector{2, Float64}(1, 1)
end

@testset "coefficient processing" begin
    x = [-2.0, -0.5, 0.0, 0.5, 2.0]
    @test threshold!(copy(x), 1.0; mode = :hard) == [-2.0, 0.0, 0.0, 0.0, 2.0]
    @test threshold!(copy(x), 1.0; mode = :soft) == [-1.0, 0.0, 0.0, 0.0, 1.0]
    @test threshold!(copy(x), 1.0; mode = :garrote) ≈ [-1.5, 0.0, 0.0, 0.0, 1.5]
    @test_throws ArgumentError threshold!(copy(x), 1.0; mode = :nope)

    y = [1.0, -5.0, 3.0, -2.0, 4.0]
    keeplargest!(y, 2)
    @test y == [0.0, -5.0, 0.0, 0.0, 4.0]
    @test count(!iszero, keeplargest!([1.0, 2.0], 0)) == 0
    @test keeplargest!([1.0, 2.0], 5) == [1.0, 2.0]
    z = collect(1.0:10.0)
    keeplargest!(z, 3)
    @test sort(z; rev = true)[1:3] == [10.0, 9.0, 8.0]

    @test noisiness(zeros(16)) == 0.0
    @test noisiness([1.0, -1.0, 1.0, -1.0]) ≈ 1 / 0.6744897501960817

    @test sparsity(fill(1.0, 8)) ≈ 0.0 atol = 1e-12
    @test sparsity([1.0; zeros(7)]) ≈ 1.0

    c = dwt(ones(8, 8), WT_D4, 1)
    e = rtenergy(c)
    @test keys(e) == (:ll, :lh, :hl, :hh)
    @test sum(values(e)) ≈ sum(abs2, c)

    x = rand(64)
    full = compress(x, WT_D4, 3; keep = 1.0)
    @test full.x ≈ x
    @test full.kept == 64
    @test full.energy ≈ 1.0

    Random.seed!(1)
    n = 256
    clean = sin.(2pi .* 4 .* range(0, 1, length = n))
    noisy = clean .+ 0.2 .* randn(n)
    plain = denoise(noisy, WT_D4, 3)
    spun = denoise(noisy, WT_D4, 3; cycles = 8)
    @test norm(plain - clean) < norm(noisy - clean)
    @test norm(spun - clean) < norm(plain - clean)
end

@testset "cascade algorithm" begin
    φ, ψ = cascade(WT_HAAR, 4)
    @test length(φ) == length(ψ) == 31
    @test all(φ[1:16] .== 1)
    @test all(φ[17:31] .== 0)
    @test sum(φ) / 2^4 ≈ 1

    φ4, ψ4 = cascade(WT_D2, 8)
    @test sum(φ4) / 2^8 ≈ 1
    @test sum(ψ4) ≈ 0 atol = 1e-8

    s3 = (1 + sqrt(3)) / 2
    φ16, _ = cascade(WT_D2, 16)
    @test [φ16[1 + k * 2^16] for k in 0:3] ≈ [0, s3, 1 - s3, 0] atol = 1e-2

    @test_throws ArgumentError cascade(WT_HAAR, 0)
end

@testset "Aqua" begin
    Aqua.test_stale_deps(NDWaveletTransforms)
    Aqua.test_deps_compat(NDWaveletTransforms)
    Aqua.test_ambiguities(NDWaveletTransforms)
    Aqua.test_piracies(NDWaveletTransforms)
end

include("view-transforms.jl")
