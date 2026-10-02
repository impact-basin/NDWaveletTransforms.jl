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

function test_image(n)
    x = range(-1, 1, length = n)
    return [0.5 + 0.4 * cos(3 * xi) * exp(-(yi^2)) + 0.25 * ((xi^2 + yi^2) < 0.25)
            for yi in x, xi in x]
end

function test_signal(n)
    t = range(0, 1, length = n)
    return collect(t), sin.(2pi .* 6 .* t) .+ 0.35 .* sin.(2pi .* 45 .* t) .+ 0.05 .* randn(n)
end

# log-magnitude of a coefficient array, for display
logmag(a) = log10.(abs.(a) .+ 1e-6)

# an image with a horizontal-striped half, a vertical-striped half and a disc
function directional_image(n)
    x = range(-1, 1, length = n)
    return [begin
        base = xi < 0 ? 0.5 + 0.4 * sin(20 * yi) : 0.5 + 0.4 * sin(20 * xi)
        xi^2 + yi^2 < 0.05 ? 1.0 : base
    end for yi in x, xi in x]
end

# --- the canonical tiling -------------------------------------------------
img = test_image(256)
fig = Figure(size = (900, 440))
ax1 = Axis(fig[1, 1]; title = "signal", yreversed = true, aspect = DataAspect())
heatmap!(ax1, img; colormap = [:black, :white])
hidedecorations!(ax1); hidespines!(ax1)

coef = dwt(img, WT_D4, 3)
ax2 = Axis(fig[1, 2]; title = "three-level D4 coefficients", yreversed = true, aspect = DataAspect())
heatmap!(ax2, logmag(coef); colormap = :viridis)
for f in (1 / 2, 1 / 4, 1 / 8)
    vlines!(ax2, [256f + 0.5]; color = (:white, 0.8), linewidth = 1)
    hlines!(ax2, [256f + 0.5]; color = (:white, 0.8), linewidth = 1)
end
for (band, (px, py)) in zip(("LL", "LH", "HL", "HH"), ((64, 64), (192, 64), (64, 192), (192, 192)))
    text!(ax2, px, py; text = band, color = :white, align = (:center, :center), fontsize = 14)
end
hidedecorations!(ax2); hidespines!(ax2)
save(joinpath(ASSETS, "tiling.png"), fig)

# --- one-dimensional coefficients, labelled by the band tree --------------
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
labels = ["LLLL", "LLLH", "LLH", "LH", "H"]
for i in 1:5
    mid = (edges[i] + edges[i + 1] - 1) / 2
    x = i == 1 ? 2 : mid
    text!(ax2, x, maximum(mag) * 0.85; text = labels[i], align = (i == 1 ? :left : :center, :center), fontsize = 12)
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
    heatmap!(axl, logmag(dwt(img, WT_D4, l)); colormap = :viridis)
    hidedecorations!(axl); hidespines!(axl)
end
save(joinpath(ASSETS, "levels.png"), fig)

# --- the four transform families ------------------------------------------
fig = Figure(size = (900, 900))
families = (("dwt", a -> dwt(a, WT_D4, 2)),
            ("wpt", a -> wpt(a, WT_D4, 2)),
            ("nsdwt", a -> nsdwt(a, WT_D4, (2, 2))),
            ("nswpt", a -> nswpt(a, WT_D4, (2, 2))))
for (i, (name, fun)) in enumerate(families)
    axf = Axis(fig[div(i - 1, 2) + 1, mod(i - 1, 2) + 1]; title = name, yreversed = true, aspect = DataAspect())
    heatmap!(axf, logmag(fun(img)); colormap = :viridis)
    hidedecorations!(axf); hidespines!(axf)
end
save(joinpath(ASSETS, "families.png"), fig)

# --- phase conventions ----------------------------------------------------
n = 256
t, s = test_signal(n)
al = dwt(s, WT_D8, 1; convention = :aligned)
wv = dwt(s, WT_D8, 1; convention = :wavelets)
fig = Figure(size = (1100, 440))
ax1 = Axis(fig[1, 1]; title = "level-1 detail, D8", xlabel = "index", ylabel = "coefficient")
half = n >> 1
lines!(ax1, 1:half, al[half+1:end]; label = ":aligned", linewidth = 2)
lines!(ax1, 1:half, wv[half+1:end]; label = ":wavelets", linewidth = 2, linestyle = :dash)
axislegend(ax1; position = :rt)
ax2 = Axis(fig[1, 2]; title = "difference of 2-D coefficients", yreversed = true, aspect = DataAspect())
diff = dwt(img, WT_D4, 1; convention = :aligned) - dwt(img, WT_D4, 1; convention = :wavelets)
heatmap!(ax2, logmag(diff); colormap = :magma)
hidedecorations!(ax2); hidespines!(ax2)
save(joinpath(ASSETS, "conventions.png"), fig)

# --- filtering in the wavelet domain --------------------------------------
function keep_bands(c, b, bands)
    k = copy(c)
    mask = falses(size(k))
    for band in bands
        rtree_view(mask, band) .= true
    end
    k[.!mask] .= 0
    return idwt(k, b, 1)
end

dimg = directional_image(256)
c1 = dwt(dimg, WT_D4, 1)
fig = Figure(size = (900, 900))
panels = (("original", dimg),
          ("LL only", keep_bands(c1, WT_D4, (:ll,))),
          ("LL + LH", keep_bands(c1, WT_D4, (:ll, :lh))),
          ("LL + HL", keep_bands(c1, WT_D4, (:ll, :hl))))
for (i, (title, im)) in enumerate(panels)
    axv = Axis(fig[div(i - 1, 2) + 1, mod(i - 1, 2) + 1]; title = title, yreversed = true, aspect = DataAspect())
    heatmap!(axv, im; colormap = [:black, :white])
    hidedecorations!(axv); hidespines!(axv)
end
save(joinpath(ASSETS, "views-filter.png"), fig)

# --- basis comparison -----------------------------------------------------
t, s = test_signal(1024)
fig = Figure(size = (1100, 450))
ax1 = Axis(fig[1, 1]; title = "coefficient decay", xlabel = "rank", ylabel = "|coefficient|", yscale = log10)
for (name, b) in (("Haar", WT_HAAR), ("D4", WT_D4), ("Sym4", WT_SYM4), ("D8", WT_D8))
    c = sort(abs.(dwt(s, b, 5)); rev = true)
    lines!(ax1, 1:length(c), max.(c, 1e-12); label = name, linewidth = 2)
end
axislegend(ax1; position = :rt)
ax2 = Axis(fig[1, 2]; title = "5% reconstruction", xlabel = "t")
lines!(ax2, t, s; color = (:gray, 0.8), linewidth = 3, label = "signal")
for (name, b, color) in (("Haar", WT_HAAR, :red), ("Sym4", WT_SYM4, :blue))
    c = dwt(s, b, 5)
    m = sort(abs.(c); rev = true)
    k = round(Int, 0.05 * length(c))
    cut = copy(c)
    cut[abs.(cut) .< m[k]] .= 0
    lines!(ax2, t, idwt(cut, b, 5); label = name, linewidth = 1.5, color = color)
end
axislegend(ax2; position = :rb)
save(joinpath(ASSETS, "bases.png"), fig)

fig = Figure(size = (900, 320))
for (i, (name, b)) in enumerate((("Haar", WT_HAAR), ("D4", WT_D4), ("Sym4", WT_SYM4)))
    axb = Axis(fig[1, i]; title = name, yreversed = true, aspect = DataAspect())
    heatmap!(axb, logmag(dwt(img, b, 2)); colormap = :viridis)
    hidedecorations!(axb); hidespines!(axb)
end
save(joinpath(ASSETS, "bases-image.png"), fig)

# --- scaling and wavelet functions ----------------------------------------
bases = (("WT_HAAR", WT_HAAR), ("WT_D4", WT_D4), ("WT_SYM4", WT_SYM4), ("WT_D8", WT_D8))
functions = [(name, cascade(b, 8)...) for (name, b) in bases]
# one symmetric range for every panel, so φ and ψ are directly comparable
m = 1.05 * maximum(max(maximum(abs, φ), maximum(abs, ψ)) for (_, φ, ψ) in functions)
fig = Figure(size = (1300, 600))
for (i, (name, φ, ψ)) in enumerate(functions)
    tt = (0:length(φ) - 1) ./ 2^8
    axp = Axis(fig[1, i]; title = name, xlabel = "t", ylabel = i == 1 ? "φ" : "")
    ylims!(axp, -m, m)
    lines!(axp, tt, φ; linewidth = 2)
    axq = Axis(fig[2, i]; xlabel = "t", ylabel = i == 1 ? "ψ" : "")
    ylims!(axq, -m, m)
    lines!(axq, tt, ψ; linewidth = 2, color = :red)
end
save(joinpath(ASSETS, "bases-functions.png"), fig)

# --- cascade convergence --------------------------------------------------
fig = Figure(size = (1000, 620))
for (i, it) in enumerate((1, 2, 3, 4, 6, 8))
    φ, _ = cascade(WT_D4, it)
    tt = (0:length(φ) - 1) ./ 2^it
    axc = Axis(fig[div(i - 1, 3) + 1, mod(i - 1, 3) + 1]; title = "n = $it", xlabel = "t")
    ylims!(axc, -1.5, 1.5)
    lines!(axc, tt, φ; linewidth = 2)
end
save(joinpath(ASSETS, "cascade-convergence.png"), fig)

# --- phase convention and the wavelet function ----------------------------
φc, ψc = cascade(WT_D4, 8)
ntaps = length(WT_D4.φ)
δ = (ntaps - 2) / 2                    # coefficient positions, (ntaps - 2) / 2
uu = (0:length(φc) - 1) ./ 2^9          # coefficient positions (half the sample grid)
fig = Figure(size = (1100, 800))
axf = Axis(fig[1, 1]; title = "scaling function φ (identical in both)", xlabel = "coefficient position")
lines!(axf, uu, φc; linewidth = 2)
axg = Axis(fig[1, 2]; title = "wavelet function ψ", xlabel = "coefficient position")
lines!(axg, uu, ψc; linewidth = 3, label = ":aligned")
lines!(axg, uu .+ δ, ψc; linewidth = 1.5, linestyle = :dash, label = ":wavelets")
axislegend(axg; position = :rt)
axh = Axis(fig[2, 1:2]; title = "detail coefficients of an impulse", xlabel = "coefficient index")
xd = zeros(256); xd[128] = 1.0
al = dwt(xd, WT_D4, 1; convention = :aligned)[129:end]
wv = dwt(xd, WT_D4, 1; convention = :wavelets)[129:end]
lines!(axh, 1:length(al), al; linewidth = 2, label = ":aligned")
lines!(axh, 1:length(wv), wv; linewidth = 2, linestyle = :dash, label = ":wavelets")
axislegend(axh; position = :rt)
save(joinpath(ASSETS, "phase-wavelets.png"), fig)

# --- cycle spinning -------------------------------------------------------
function denoise!(x)
    dwt!(x, WT_D4, 4)
    thr = 0.1 * maximum(abs.(x))
    x[abs.(x) .< thr] .= 0
    idwt!(x, WT_D4, 4)
    return x
end

n = 512
t = range(0, 1, length = n)
clean = [xi < 0.5 ? 1.0 : 0.0 for xi in t]
noisy = clean .+ 0.15 .* randn(n)
plain = denoise!(copy(noisy))
spun = copy(noisy)
cyclespinning!(denoise!, spun, 8)

fig = Figure(size = (900, 480))
ax = Axis(fig[1, 1]; title = "cycle spinning", xlabel = "t", ylabel = "signal")
lines!(ax, collect(t), noisy; color = (:gray, 0.35), label = "noisy")
lines!(ax, collect(t), clean; color = :black, linewidth = 2, label = "clean")
lines!(ax, collect(t), plain; color = :red, label = "thresholded")
lines!(ax, collect(t), spun; color = :blue, linewidth = 1.5, label = "spin-cycled")
axislegend(ax; position = :rb, nbanks = 2)
save(joinpath(ASSETS, "spinning.png"), fig)

println("figures written to ", ASSETS)
