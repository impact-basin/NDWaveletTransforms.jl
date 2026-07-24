using Test
using Waiver

@testset "Haar transforms" begin
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

# y = copy(x)
# @time nsdwt!(x, WT_D4, (2, 3));
# @time nsidwt!(x, WT_D4, (2, 3));
# @time nsdwt!(x, WT_HAAR, (2, 3));
# @time nsidwt!(x, WT_HAAR, (2, 3));
#
# f = Figure()
# a, b, c = Axis.([f[1,1], f[1,2], f[1,3]])
# heatmap!(a, x)
# heatmap!(b, x)
# h = heatmap!(c, x ./ y)
# Colorbar(f[1,4], h)
# x ≈ y
