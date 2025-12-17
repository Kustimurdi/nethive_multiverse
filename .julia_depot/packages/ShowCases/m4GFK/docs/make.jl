using ShowCases
using Documenter

DocMeta.setdocmeta!(ShowCases, :DocTestSetup, :(using ShowCases); recursive=true)

makedocs(;
    modules=[ShowCases],
    authors="Expanding Man <savastio@protonmail.com> and contributors",
    repo="https://gitlab.com/ExpandingMan/ShowCases.jl/blob/{commit}{path}#{line}",
    sitename="ShowCases.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ExpandingMan.gitlab.io/ShowCases.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md",
    ],
)
