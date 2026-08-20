using Test
using NDWaveletTransforms

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

@testset "rtree indexing: lengths OK" begin
    x = rand(4)
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
