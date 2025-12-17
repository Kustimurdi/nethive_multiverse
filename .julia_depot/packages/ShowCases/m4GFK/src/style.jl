
"""
    Style

A struct containing metadata for specifying the "style" (i.e. color and other attributes used when
printing to some displays) of an object to be shown.

Styles can be used in `show` or `print` with `show(io, 𝔰, x)` or `print(io, 𝔰, x)` where `𝔰` is a
`Style` and `x` is any other object.

## Constructors
- `Style(color, bold)`
- `Style(;color=:normal, bold=false)`

## Arguments
- `color`: The color to print or show with as it would be provided to `printstyled`.  See `printstyled`
    documentation for allowed color specifiers (these are either `Symbol` e.g. `:red` or integers).
- `bold`: A boolean specifying whether the object should be shown with bold text.
"""
struct Style
    none::Bool  # special flag to indicate use of regular `print`
    color::Color
    bold::Bool

    # would be nice to eventually add more options here
end

function Style(color::Color, bold::Bool=false)
    none = (color == :normal && !bold)
    Style(none, color, bold)
end
Style(;color::Color=:normal, bold::Bool=false) = Style(color, bold)

function withstyle(𝒻, io::IO, 𝔰::Style, x...)
    𝔰.none ? 𝒻(io, x...) : Base.with_output_color(𝒻, 𝔰.color, io, x...; bold=𝔰.bold)
end

Base.show(io::IO, 𝔰::Style, x) = withstyle(show, io, 𝔰, x)
Base.print(io::IO, 𝔰::Style, x...) = withstyle(print, io, 𝔰, x...)


"""
    Styled{𝒯}

An `AbstractShow` wrapper which when shown will be shown as the wrapped object but with `Style`
such as color or text boldness.

## Constructors
- `Styled(o, color, bold=false)`
- `Styled(o; color=:normal, bold=false)`

## Arguments
- `o`: The object to be wrapped.
- `color`: A color specifier as it would be passed to `prinstyled`, see also `Style`.
- `bold`: Whether text should be rendered as bold.
"""
struct Styled{𝒯} <: AbstractShow{𝒯}
    object::𝒯
    style::Style
end

Styled(o, color::Color, bold::Bool=false) = Styled(o, Style(color, bold))
Styled(o; kw...) = Styled(o, Style(;kw...))

Base.show(io::IO, s::Styled) = show(io, s.style, s.object)
Base.print(io::IO, s::Styled) = print(io, s.style, s.object)

