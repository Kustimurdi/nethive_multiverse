
"""
    ShowIndents

An `AbstractShow` wrapper which inserts indents into each line in the shown output of the object it wraps.

## Constructors
- `ShowIndents(o, indent="    ")`

## Arguments
- `o`: the object to be wrapped.
- `indent::AbstractString`: the indent to insert.
"""
struct ShowIndents{𝒯} <: AbstractShow{𝒯}
    object::𝒯
    indent::String
end

ShowIndents(o, indent="    ") = ShowIndents(o, indent)

function with(𝒻, io::IO, s::ShowIndents)
    b = IOContext(IOBuffer(), io)
    𝒻(b, s.object)
    seekstart(b.io)
    ind = countlines(b.io) > 1
    seekstart(b.io)
    if ind
        for l ∈ eachline(b.io, keep=true)
            print(io, s.indent)
            write(io, l)
        end
    else
        write(io, take!(b.io))
    end
    nothing
end

Base.show(io::IO, s::ShowIndents) = with(show, io, s)
Base.print(io::IO, s::ShowIndents) = with(show, io, s)


struct ShowString{𝒯} <: AbstractShow{𝒯}
    object::𝒯
    block::Symbol
    indent::String
end

function ShowString(o; block::Symbol=:context, indent::String="")
    ShowString(ShowIndents(Print(o), indent), block, indent)
end

function with(𝒻, io::IO, s::ShowString)
    b = IOContext(IOBuffer(), io)
    𝒻(b, s.object)
    seekstart(b.io)
    block = s.block == :always || (s.block ≠ :never && countlines(b.io) > 1)
    block ? print(io, "\"\"\"\n") : print(io, "\"")
    write(io, take!(b.io))
    block ? print(io, "$(s.indent)\"\"\"") : print(io, "\"")
end

Base.show(io::IO, s::ShowString) = with(show, io, s)
Base.print(io::IO, s::ShowString) = with(print, io, s)


struct ShowLimit{𝒯} <: AbstractShow{𝒯}
    object::𝒯
    limit::Int  # character limit
    n::Int  # initial position
    cstr::String  # continuation string
    checkbrack::Bool
end

function ShowLimit(o; limit::Integer=typemax(Int), n::Integer=0, continuation_string::AbstractString="…",
                   check_brackets::Bool=true)
    ShowLimit(o, limit, n, continuation_string, check_brackets)
end

function Base.show(io::IO, s::ShowLimit)
    withlimit(show, io, s.object, limit=s.limit, n=s.n, continuation_string=s.cstr, check_brackets=s.checkbrack)
end
function Base.print(io::IO, s::ShowLimit)
    withlimit(print, io, s.object, limit=s.limit, n=s.n, continuation_string=s.cstr, check_brackets=s.checkbrack)
end

"""
    ShowCat(o...; kw...)

🐈 Show objects in series with no separation. `kw` are passed to [`ShowList`](@ref).

For example, `ShowCat("a", 1, 2)` will show `"a"12`.
"""
ShowCat(x...; brackets::AbstractString="", delim=Print(""), kw...) = ShowList(x; brackets, delim, kw...) # 🐈

"""
    ShowKw(o, name, separator=Print("="); kw...)
    ShowKw(o, name::Symbol, separator=Print("="); kw...)

Show an object as if it were a keyword argument with name `name`.  For example `ShowKw(x, :x)` shows
`x=x`.

## Constructors
- `o`: the object to be wrapped.
- `name`: The name of the object, if a `Symbol` will be printed as in keyword arguments.
- `separator`: separator between the name and shown object.
- `kw`: Keyword arguments passed to [`ShowList`](@ref).
"""
function ShowKw(o, keyword, separator=Print("="); kw...)
    ShowCat(keyword, separator, o; kw...)
end
ShowKw(o, name::Symbol, sep=Print("="); kw...) = ShowKw(o, Print(string(name)), sep; kw...)

"""
    ShowProps(o, props=propertynames(o);
              show_keywords=true, new_lines=false, keyword_style=Style(), entry_style=Style()
              separator=(new_lines ? Print(" = ") : Print("=")),
              kw...)

Show the properties of an object `o` via [`ShowList`](@ref) with entries using `ShowKw`.

## Arguments
- `props`: Symbols with names of properties to show.
- `show_keywords`: whether or not to show the property names.
- `keyword_style::Style`: a style to apply to the keywords.
- `entry_style::Style`: style to apply to entries.
- `separator`: Separator to use between keyword and property.  By default, spacing will be added if `new_lines`.
- `kw`: other keywords passed to [`ShowList`](@ref)

## Examples
```
julia> propertynames(1.0 + im)
(:re, :im)

julia> ShowProps(1.0 + im)
(re=1.0, im=1.0)

julia> ShowProps(1.0 + im, [:re])
(re=1.0)

julia> ShowProps(1.0 + im, [:re], keyword_style=Style(:red), new_lines=true)
(
    re = 1.0
)

julia> ShowProps(1.0 + im, show_keywords=false)
(1.0, 1.0)
```
"""
function ShowProps(o, props=propertynames(o);
                   show_keywords::Bool=true,
                   new_lines::Bool=false,
                   keyword_style::Style=Style(),
                   entry_style::Style=Style(),
                   separator=(new_lines ? Print(" = ") : Print("=")),
                   kw...)
    𝒻 = if show_keywords
        i -> ShowKw(Styled(getproperty(o, props[i]), entry_style), Styled(Print(string(props[i])), keyword_style), separator)
    else
        i -> Styled(getproperty(o, props[i]), entry_style)
    end
    tpl = ntuple(𝒻, length(props))
    ShowList(tpl; new_lines, kw...)
end

"""
    ShowCase(o, props=propertynames(o);
             type_style=Style(), show_module=:context,
             max_params=3, show_params=true, kw...)

Creates and `AbstractShow` object which combines all elements used in default `show` methods.  This concatenates
[`ShowTypeOf`](@ref) and [`ShowProps`](@ref).

## Arguments
- `o`: object to be wrapped.
- `props`: properties of the object to be shown.
- `type_style`: Style to be used for the type.
- `show_module::Symbol`: whether to show the module of type and type parameters, see [`ShowType`](@ref).
- `max_params::Integer`: maximum number of type parameters to show.
- `show_params::Bool`: whether to show params. Note that if `max_params=0` and `show_params=true`, brackets with
    continuation strings will be shown.
- `kw`: other keyword arguments passed to [`ShowProps`](@ref).

## Examples
```
julia> ShowCase(1.0 + im)
ComplexF64{Float64}(re=1.0, im=1.0)

julia> ShowCase(1.0 + im, new_lines=true)
ComplexF64{Float64}(
    re = 1.0,
    im = 1.0
)

julia> ShowCase(1.0 + im, type_style=Style(:cyan, true), max_params=0)
ComplexF64{…}(re=1.0, im=1.0)

julia> ShowCase(1.0 + im, [:im], keyword_style=Style(:magenta))
ComplexF64{Float64}(im=1.0)
```
"""
function ShowCase(o, props=propertynames(o);
                  type_style::Style=Style(), show_module::Symbol=:context,
                  max_params::Integer=3, show_params::Bool=true,
                  kw...
                 )
    ShowCat(ShowTypeOf(o; type_style, show_module, show_params, max=max_params), ShowProps(o, props; kw...))
end
