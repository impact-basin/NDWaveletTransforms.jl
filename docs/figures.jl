# figures.jl -- regenerate the figures embedded in the documentation.
#
#     julia --project=docs docs/figures.jl
using CairoMakie
using NDWaveletTransforms
using Random
using LinearAlgebra

const ASSETS = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS)

Random.seed!(20240917)

# a deterministic image with a smooth ramp, a disc and a sharp edge
function test_image(n)
    x = range(-1, 1, length = n)
    return [0.5 + 0.4 * cos(3 * xi) * exp(-(yi^2)) +
            0.25 * ((xi^2 + yi^2) < 0.25)
            for yi in x, xi in x]
end

function test_signal(n)
    t = range(0, 1, length = n)
    return collect(t), sin.(2pi .* 6 .* t) .+ 0.35 .* sin.(2pi .* 45 .* t) .+ 0.05 .* randn(n)
end

# --- the canonical tiling -------------------------------------------------
img = test_image(256)
fig = Figure(size = (900, 440))
ax1 = Axis(fig[1, 1]; title = "signal", yreversed = true, aspect = DataAspect())
heatmap!(ax1, img; colormap = [:black, :white])
hidedecorations!(ax1); hidespines!(ax1)

coef = dwt(img, WT_D4, 3)
ax2 = Axis(fig[1, 2]; title = "three-level D4 coefficients", yreversed = true, aspect = DataAspect())
heatmap!(ax2, log10.(abs.(coef) .+ 1e-6); colormap = :viridis)
for f in (1 / 2, 1 / 4, 1 / 8)
    vlines!(ax2, [256f + 0.5]; color = (:white, 0.8), linewidth = 1)
    hlines!(ax2, [256f + 0.5]; color = (:white, 0.8), linewidth = 1)
end
for (band, (px, py)) in zip(("LL", "LH", "HL", "HH"), ((64, 64), (192, 64), (64, 192), (192, 192)))
    text!(ax2, px, py; text = band, color = :white, align = (:center, :center), fontsize = 14)
end
hidedecorations!(ax2); hidespines!(ax2)
save(joinpath(ASSETS, "tiling.png"), fig)

# --- one-dimensional coefficients -----------------------------------------
t, s = test_signal(512)
coef = dwt(s, WT_D4, 4)
mag = log10.(abs.(coef) .+ 1e-8)
fig = Figure(size = (900, 520))
ax1 = Axis(fig[1, 1]; title = "signal", xlabel = "t")
lines!(ax1, t, s; color = :black, linewidth = 1)
ax2 = Axis(fig[2, 1]; title = "four-level D4 coefficients", xlabel = "index", ylabel = "log10 |coef|")
lines!(ax2, 1:length(coef), mag; color = :black, linewidth = 1)
edges = [1; reverse([512 >> l for l in 1:4]) .+ 1; 513]
for b in edges[2:end-1]
    vlines!(ax2, [b - 0.5]; color = (:gray, 0.6), linestyle = :dash)
end
labels = ["A4", "D4", "D3", "D2", "D1"]
for i in 1:5
    mid = (edges[i] + edges[i + 1] - 1) / 2
    text!(ax2, mid, maximum(mag) * 0.85; text = labels[i], align = (:center, :center), fontsize = 12)
end
save(joinpath(ASSETS, "coefficients.png"), fig)

# --- sparsity -------------------------------------------------------------
t, s = test_signal(2048)
fig = Figure(size = (700, 450))
ax = Axis(fig[1, 1]; title = "sparsity of the DWT", xlabel = "retained fraction", ylabel = "relative error")
fracs = 0.005:0.005:0.15
for (name, b) in (("Haar", WT_HAAR), ("D4", WT_D4), ("D8", WT_D8), ("D16", WT_D16))
    cf = dwt(s, b, 6)
    magnitudes = sort(abs.(cf); rev = true)
    errs = map(fracs) do f
        k = max(1, round(Int, f * length(cf)))
        cut = copy(cf)
        cut[abs.(cut) .< magnitudes[k]] .= 0
        norm(idwt(cut, b, 6) - s) / norm(s)
    end
    lines!(ax, collect(fracs), errs; label = name, linewidth = 2)
end
axislegend(ax; position = :rt)
save(joinpath(ASSETS, "compression.png"), fig)

# --- per-axis levels ------------------------------------------------------
fig = Figure(size = (900, 440))
for (i, l) in enumerate(((1, 3), (3, 1)))
    axl = Axis(fig[1, i]; title = "levels = ($(l[1]), $(l[2]))", yreversed = true, aspect = DataAspect())
    heatmap!(axl, log10.(abs.(dwt(img, WT_D4, l)) .+ 1e-6); colormap = :viridis)
    hidedecorations!(axl); hidespines!(axl)
end
save(joinpath(ASSETS, "levels.png"), fig)

println("figures written to ", ASSETS)
