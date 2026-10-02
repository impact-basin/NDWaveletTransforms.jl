using Documenter
using NDWaveletTransforms

makedocs(;
    sitename = "NDWaveletTransforms.jl",
    authors = "Henry Eshbaugh",
    modules = [NDWaveletTransforms],
    checkdocs = :none,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://impact-basin.github.io/NDWaveletTransforms.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "Phase conventions" => "conventions.md",
        "Subbands and views" => "views.md",
        "Bases" => "bases.md",
        "GPU" => "gpu.md",
        "API reference" => "api.md",
    ],
)

if get(ENV, "CI", "false") == "true"
    deploydocs(;
        repo = "github.com/impact-basin/NDWaveletTransforms.jl",
        devbranch = "master",
    )
end
