
struct ShowEntry{𝒯,𝒟} <: AbstractShow{𝒯}
    object::𝒯
    newline::Bool
    indent::String  # only printed if newline
    delim::𝒟
end

function ShowEntry(o; new_line::Bool=false, indent::AbstractString="    ",
                   delim=Print(", "))
    ShowEntry(o, new_line, indent, delim)
end
ShowEntry(o, 𝔰::Style; kw...) = ShowEntry(Styled(o, 𝔰); kw...)

function with(𝒻, io::IO, s::ShowEntry)
    if s.newline
        print(io, "\n")
        print(io, s.indent)
    end
    𝒻(io, s.object)
    𝒻(io, s.delim)
    nothing
end

Base.show(io::IO, s::ShowEntry) = with(show, io, s)
Base.print(io::IO, s::ShowEntry) = with(print, io, s)


struct ShowList{𝒯,ℬₗ,ℬᵣ,𝒞,𝒟} <: AbstractShow{𝒯}
    object::𝒯
    lbracket::ℬₗ
    rbracket::ℬᵣ
    delim::𝒟
    newlines::Bool
    indent::String
    continuation::𝒞
    entry_style::Union{Nothing,Style}
    max::Int
end

ShowEntry(o::ShowEntry, ::ShowList; kw...) = o
function ShowEntry(o, l::ShowList; is_last::Bool=false)
    o = isnothing(l.entry_style) ? o : Styled(o, l.entry_style)
    ShowEntry(o; new_line=l.newlines, indent=l.indent, delim=(is_last ? Print("") : l.delim))
end

Base.length(l::ShowList) = length(l.object)
Base.getindex(l::ShowList, j)  = ShowEntry(l.object[j], l, is_last=(j == length(l)))

Base.IteratorSize(::Type{<:ShowList{𝒯}}) where {𝒯} = Base.IteratorSize(𝒯)
Base.IteratorEltype(::Type{<:ShowList{𝒯}}) where {𝒯} = Base.IteratorEltype(𝒯)

function Base.iterate(l::ShowList)
    xj = iterate(l.o)
    isnothing(xj) && return nothing
    ShowEntry(xj[1], l), xj[2]
end
function Base.iterate(l::ShowList, j)
    xj = iterate(l.o, j)
    isnothing(xj) && return nothing
    ShowEntry(xj[1], l), xj[2]
end

_bracket_wrap_func(::Nothing) = Print
_bracket_wrap_func(𝔰::Style) = o -> Styled(o, 𝔰)
_get_left_bracket(b::AbstractString, 𝔰)  = _bracket_wrap_func(𝔰)(isempty(b) ? "" : string(collect(b)[1]))
_get_right_bracket(b::AbstractString, 𝔰) = _bracket_wrap_func(𝔰)(isempty(b) ? "" : string(collect(b)[2]))

function ShowList(o;
                  brackets::AbstractString="()",
                  bracket_style::Union{Style,Nothing}=nothing,
                  left_bracket=_get_left_bracket(brackets, bracket_style),
                  right_bracket=_get_right_bracket(brackets, bracket_style),
                  new_lines::Bool=false, indent::AbstractString="    ",
                  delim=Print(", "), max::Integer=8,
                  continuation=Print("…"), entry_style::Union{Nothing,Style}=nothing
                 )
    ShowList(o, left_bracket, right_bracket, delim, new_lines, indent, continuation, entry_style, max)
end
ShowList(o...; kw...) = ShowList(o; kw...)

# TODO for now we only work for known length, because unknown length is a huge pain in the ass
function with(𝒻, io::IO, s::ShowList)
    𝒻(io, s.lbracket)
    m = min(s.max, length(s.object))
    for i ∈ 1:m
        𝒻(io, s[i])
    end
    m < length(s.object) && 𝒻(io, s.continuation)
    s.newlines && print(io, "\n")
    𝒻(io, s.rbracket)
    nothing
end

Base.show(io::IO, s::ShowList) = with(show, io, s)
Base.print(io::IO, s::ShowList) = with(print, io, s)


