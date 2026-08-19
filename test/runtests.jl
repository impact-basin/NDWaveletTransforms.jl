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
