# Regression tests for transforms on strided subband views.
#
# dwt! allocates similar(x) as scratch. For a view that buffer is dense while
# the view is strided, and the N-D axis passes indexed the output with the
# input's strides, writing past the end of the buffer. The eager tests below
# reconstruct the view and check that coefficients outside it never move; on
# the pre-fix tree they abort the process rather than failing a test.
#
# Not wired into runtests.jl until the stage 1 stride fix lands.

using Test
using Random
using NDWaveletTransforms

const VIEW_RNG = MersenneTwister(20240917)

# transform the LL subband of a one-level transform in place, compare against
# a dense reference, and confirm the other subbands are untouched.
function view_case(T, dims, extra, convention)
    rng = VIEW_RNG
    x = rand(rng, T, dims)
    dwt!(x, WT_HAAR, 1; convention = convention)

    before = copy(x)
    ll = rtree_view(x, 1)
    ref = copy(ll)
    outside = trues(size(x))
    rtree_view(outside, 1) .= false

    dwt!(ll, WT_HAAR, extra; convention = convention)
    @test ll ≈ dwt(ref, WT_HAAR, extra; convention = convention)
    @test x[outside] == before[outside]

    idwt!(ll, WT_HAAR, extra; convention = convention)
    @test ll ≈ ref
    return nothing
end

@testset "strided subband views" begin
    for T in (Float32, Float64), convention in (:aligned, :wavelets)
        @testset "$T, $convention" begin
            view_case(T, 64, 2, convention)
            view_case(T, (64, 64), 2, convention)
            view_case(T, (16, 16, 16), 2, convention)
        end
    end
end

@testset "@rtview in place" begin
    rng = VIEW_RNG
    x = rand(rng, 32, 32)
    dwt!(x, WT_HAAR, 1)

    before = copy(x)
    ref = copy(@rtview x[:ll])
    outside = trues(size(x))
    rtree_view(outside, :ll) .= false

    dwt!(@rtview(x[:ll]), WT_HAAR, 2)
    @test @rtview(x[:ll]) ≈ dwt(ref, WT_HAAR, 2)
    @test x[outside] == before[outside]

    idwt!(@rtview(x[:ll]), WT_HAAR, 2)
    @test @rtview(x[:ll]) ≈ ref
end

@testset "transforms on underscore views" begin
    rng = VIEW_RNG
    x = rand(rng, 32, 32)
    dwt!(x, WT_HAAR, (1, 2))

    # the (1, 2) transform is two levels deep on axis 2, one on axis 1
    v = @rtview x[:ll, :_l]
    @test size(v) == (16, 8)

    ref = copy(v)
    dwt!(v, WT_HAAR, 2)
    @test v ≈ dwt(ref, WT_HAAR, 2)
    idwt!(v, WT_HAAR, 2)
    @test v ≈ ref
end
