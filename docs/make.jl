using RepLieGroups
using Documenter

DocMeta.setdocmeta!(RepLieGroups, :DocTestSetup, :(using RepLieGroups); recursive=true)

makedocs(;
    modules=[RepLieGroups],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/RepLieGroups.jl/blob/{commit}{path}#{line}",
    sitename="RepLieGroups.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/RepLieGroups.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/RepLieGroups.jl",
    devbranch="main",
)
