using Test
using Waiver

@testset "Haar traforms" begin
    x = testimage("moonsurface") .|> Float32
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
    x = testimage("moonsurface") .|> Float32
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
    x = testimage("moonsurface") .|> Float32
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
    x = testimage("moonsurface") .|> Float32
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
    x = testimage("moonsurface")
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
    x = testimage("moonsurface")
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

@testset "resolution test 1920" begin
    x = testimage("resolution_test_1920")
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

@testset "resolution test 1920, nonstandard" begin
    x = testimage("resolution_test_1920")
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
