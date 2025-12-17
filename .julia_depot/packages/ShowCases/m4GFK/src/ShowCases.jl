module ShowCases


# may ultimately do something a little less silly in which case we'll need to substitute this
const Color = Union{Int,Symbol}


function _print_bracket_close(io::IO, s::AbstractString)
    for lr ∈ ("()", "[]", "{}", "''", "\"\"", "``")
        startswith(s, lr[1]) && print(io, lr[2])
    end
end

"""
    withlimit(𝒻, io::IO, args...; limit=typemax(Int), n=0, continuation_string="…")

Executes `𝒻(io, args...)` returning the number of *characters* (not bytes) written up to the character limit `limit`.
If the limit is reached, `continuation_string` will be printed, but does not count toward the limit.  `n` can be
used to provide a starting position when counting toward the limit.
"""
function withlimit(𝒻, io::IO, args...; limit::Integer=typemax(Int), n::Integer=0, continuation_string::AbstractString="…",
                   check_brackets::Bool=true)
    b = IOContext(IOBuffer(), io)
    𝒻(b, args...)
    s = String(take!(b.io))
    if limit ≥ length(s) + n
        write(io, s)
    else
        o = write(io, view(s, 1:(limit - n)))
        # we deliberately exclude styling from this, though unfortunately it's rather inflexible
        print(io, continuation_string)  # we ignore these for positioning
        check_brackets && _print_bracket_close(io, s)
        o
    end + n
end


"""
    AbstractShow{𝒯}

The abstract type of objects to be shown in the `ShowCases` module.  All such objects
wrap a single object of type `𝒯` accessible via the `object` field name.

Objects of the `AbstractShow` type can be provided to the multi-argument `show` method,
e.g. `show(io, Show(ξ), Print(" + "), Show(ζ))`.
"""
abstract type AbstractShow{𝒯} end

objecttype(o) = typeof(o)
objecttype(s::AbstractShow) = objecttype(s.object)

Base.show(io::IO, s::AbstractShow...) = foreach(σ -> show(io, σ), s)
Base.show(s::AbstractShow...) = show(stdout, s...)


"""
    ShowNothing

A type which shows nothing.  Neither `show(io, ShowNothing())` and `print(io, ShowNothing())`
will insert anything into the stream.  This can be useful as a placeholder in some other functions
in the `ShowCases` module.
"""
struct ShowNothing <: AbstractShow{Nothing} end

Base.show(::IO, ::ShowNothing) = nothing
Base.print(::IO, ::ShowNothing) = nothing

"""
    Show

An `AbstractShow` wrapper of another object.  This is useful for methods which require
`AbstractShow` objects, for example the multi-argument `Base.show` method.

## Constructors
- `Show(x)`; where `x` is the object to be wrapped.
"""
struct Show{𝒯} <: AbstractShow{𝒯}
    object::𝒯
end

Base.show(io::IO, s::Show) = show(io, s.object)
Base.print(io::IO, s::Show) = print(io, s.object)

"""
    Print

An `AbstractShow` wrapper that forces the underlying object to be shown using the
`print` method.  That is, calling `show(io, Print(x))` is equivalent to `print(io, x)`.
This is particularly useful for passing arbitrary strings to functions in the `ShowCases`
module.

## Constructors
- `Print(x)`; where `x` is the object to be wrapped.

## Examples
```
julia> Print("spock")
spock

```
Note the lack of printed `"` characters.
"""
struct Print{𝒯} <: AbstractShow{𝒯}
    object::𝒯
end

Base.show(io::IO, p::Print) = print(io, p)
Base.print(io::IO, p::Print) = print(io, p.object)


include("style.jl")
include("list.jl")
include("types.jl")
include("misc.jl")
include("docs.jl")


export Show, ShowLimit, Style, Styled, Print, ShowEntry, ShowList, ShowType, ShowBaseType, ShowParams, ShowCat,
    ShowKw, ShowProps, ShowNothing, ShowCase, ShowTypeOfBase, ShowTypeOf, ShowIndents, ShowString

end
