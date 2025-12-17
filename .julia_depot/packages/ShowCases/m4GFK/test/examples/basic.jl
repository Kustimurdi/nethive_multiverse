using ShowCases


struct BasicExample{𝒯}
    x::Int
    y::Float64
    z::Vector{𝒯}
    s::String
end

basic_example() = BasicExample{ComplexF64}(0, 1.0, [1.0, -1.0*im], "example")

# verbose, multiple lines with type in bold
function show1(io::IO, b::BasicExample)
    s = ShowCase(b, new_lines=true, type_style=Style(:cyan, true))
    show(io, s)
end

# always show modules, only show `y` and `s` properties
function show2(io::IO, b::BasicExample)
    s = ShowCase(b, [:y, :s], show_module=:always)
    show(io, s)
end

# an extremely terse version that only displays the property `x`
function show3(io::IO, b::BasicExample)
    s = ShowCase(b, [:x], show_module=:never, show_params=false, show_keywords=false)
    show(io, s)
end

# a weird example that makes a few customizations, including contextual color of `x` property
function show4(io::IO, b::BasicExample)
    l = ShowList(Styled(ShowKw(b.x, :x, Print(": ")), b.x ≥ 0.0 ? :blue : :red),
                 Show(b.y), new_lines=true)
    show(io, ShowTypeOf(b), l)
end
