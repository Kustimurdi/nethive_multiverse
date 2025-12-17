# ShowCases

[![dev](https://img.shields.io/badge/docs-latest-blue?style=for-the-badge&logo=julia)](https://ExpandingMan.gitlab.io/ShowCases.jl/)
[![build](https://img.shields.io/gitlab/pipeline/ExpandingMan/ShowCases.jl/master?style=for-the-badge)](https://gitlab.com/ExpandingMan/ShowCases.jl/-/pipelines)

A Julia package for easily declaring `show` methods with pretty output.

## Examples
```julia
struct BasicExample{𝒯}
    x::Int
    y::Float64
    z::Vector{𝒯}
    s::String
end
```

**De-cluttering Example:**
```julia
ShowCase(x, new_lines=true, type_style=Style(:cyan, true))
```
[![basic_example_1](docs/images/basic_example_1.png)](https://gitlab.com/ExpandingMan/ShowCases.jl/-/blob/master/test/examples/basic.jl)

**Customization Example:**
```julia
l = ShowList(Styled(ShowKw(x.x, :x, Print(": ")), b.x ≥ 0.0 ? :blue : :red), Show(b.y),
             new_lines=true)
show(io, ShowTypeOf(x), l)
```
[![basic_example_4](docs/images/basic_example_4.png)](https://gitlab.com/ExpandingMan/ShowCases.jl/-/blob/master/test/examples/basic.jl)


**Complex Example:**

[![message_example](docs/images/message_example.png)](https://gitlab.com/ExpandingMan/ShowCases.jl/-/blob/master/test/examples/message.jl)
