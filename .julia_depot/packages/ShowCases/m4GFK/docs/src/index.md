```@meta
CurrentModule = ShowCases
```

# ShowCases

This is a module for simplifying the customization of Julia's `show` methods.  If your
module defines new types and you are unsatisfied with the output of the default `show`
method, you may find this module useful.


## Introduction
By default, Julia calls `show(io, x)` to display the object `x` e.g. in the REPL.  By default, new
objects receive a defualt method which prints the default constructor with the arguments
contained in the object.  For example, a struct `TestStruct` with parameter `𝒯=Int` and
fields `x = 0` and `y = 1` would appear as `TestStruct{Int}(0, 1)`.  This is a very
reasonable default, but for more complicated types its output may be obnoxious.  At the
very least, it becomes likely that the default `show` output is too verbose.  This can be
solved by defining a custom `show` method, however, there aren't any specialized tools in
`Base` for doing this nicely so one must build up the custom method with `print` and
`show` calls of the object and its constituents.  This is good enough for simple displays,
but for more elaborate objects it can be laborious.

This module attempts to solve that by providing the user with a set of composable wrapper
objects which, when displayed with `show` or `print` have their own behavior.  From this
one can build fairly elaborate `show` functions with relatively little effort.

!!! note

    This package was designed specifically for creating `show` methods which display
    objects in ways which at least vaguely resemble Julia syntax.  This is *NOT* a package
    for showing specialized data structures such as trees or tables in pictorial or
    "graphical" representations.  While we would like to support the ability to embed such
    elaborate representations in `ShowCases` objects, the variety of such representations
    makes it very hard to guarantee that even this will work nicely in the general case.
    See [AbstractTrees](https://github.com/JuliaCollections/AbstractTrees.jl) or
    [PrettyTables](https://github.com/ronisbr/PrettyTables.jl) if you are looking for help
    showing trees or tables.


## Basic Examples
We declare the struct
```julia
struct BasicExample{𝒯}
    x::Int
    y::Float64
    z::Vector{𝒯}
    s::String
end
```

The default `show` method for this object will show all fields, is rather verbose, and
arguably inelegantly jumbled together.

We can use `ShowCases` to customize the `show` output relatively easily.  The `ShowCase`
method shows all of the information normally shown by the fall-back method for `show` with
slightly different defaults.
```julia
b = BasicExample{ComplexF64}(0, 1.0, [1.0, -1.0*im], "example")

s = ShowCase(b)  # object to be shown
```

If we then do `show(b)` we get:
```
BasicExample{ComplexF64}(x=0, y=1.0, z=ComplexF64[1.0 + 0.0im, -0.0 - 1.0im], s="example")
```
In this case the only difference from the default method is the addition of keywords.

We can use the options build into `ShowCase` and others to get a few nicer ways of showing
this.
```julia
julia> s = ShowCase(b, new_lines=true)
BasicExample{ComplexF64}(
    x = 0,
    y = 1.0,
    z = ComplexF64[1.0 + 0.0im, -0.0 - 1.0im],
    s = "example"
)

julia> ShowCase(b, new_lines=true)
BasicExample{ComplexF64}(
    x = 0,
    y = 1.0,
    z = ComplexF64[1.0 + 0.0im, -0.0 - 1.0im],
    s = "example"
)

julia> ShowCase(b, new_lines=true, max_params=0)
BasicExample{…}(
    x = 0,
    y = 1.0,
    z = ComplexF64[1.0 + 0.0im, -0.0 - 1.0im],
    s = "example"
)

julia> ShowCase(b, [:x, :y], show_keywords=false)
BasicExample{ComplexF64}(0, 1.0)

julia> ShowCase(b, [:s], show_params=false)
BasicExample(s="example")
```

Calling [`ShowCases.ShowCase`](@ref) will create an object the `show` method of which will reflect the
arguments we have chosen.  It is composed of other, similar objects, which can be used
show objects with even more customization.
```julia
julia> show(ShowTypeOf(b, show_params=false), ShowList(Styled(b.x, :blue), b.y, brackets="[]"))
BasicExample[0, 1.0]

julia> show(Show(b.s), ShowProps(b, [:x, :z], brackets="{}"))
"example"{x=0, z=ComplexF64[1.0 + 0.0im, -0.0 - 1.0im]}
```
`show` accepts multiple arguments if they are of the `AbstractShow` type.


## Composing Objects
All of the `AbstractShow` objects provided by `ShowCases` are intended to be composable.
With this we can get more complex behavior by wrapping one `AbstractShow` in another, for
example
```julia
# shows characters up to a certain limit, checking for brackets and closing them
julia> l = ShowLimit("abcdef", limit=3)
"ab…"

# print with a style on displays which can support it
julia> 𝔰 = Styled(l, :red)
"ab…"

# print as a list entry; in this case just adds a `,`
julia> e = ShowEntry(𝔰)
"ab…",

# ShowEntry is treated specially by ShowList to override defaults
julia> ShowList(e, 2, 3, delim=Print("; "))
("ab…", 2; 3)
```


## Use Cases
The most common use case for `ShowCases` is expected to be in defining `show` methods.  In
Julia, by default objects are displayed (e.g. in the REPL) using `show(stdout, o)`.
Therefore if one defines some type `Type1`, defining `show(::IO, ::Type1)` can be used to
set custom `show` behavior.  Typically using `ShowCases` would involve defining `show`
methods which construct `AbstractShow` objects and insert them into the stream in `show`,
for example
```julia
Base.show(io::IO, t::Type1) = show(io, ShowCase(t))
```

In other cases, `ShowCases` may be useful even without defining show methods.  For
example, if you do not want to override the existing `show` behavior, or you are working
with objects defined in other packages, you may want to use `ShowCases` to view objects in
a particular way that you commonly need.
