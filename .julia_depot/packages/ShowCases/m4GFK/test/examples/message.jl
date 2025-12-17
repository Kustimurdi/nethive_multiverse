using ShowCases


struct Message
    headers::Dict{String,String}
    body::Vector{UInt8}
end

function message_example()
    body = """
    This is a demonstration
    of the
    ShowCases.jl
    package.
    """
    Message(Dict(":method"=>"GET", ":path"=>"/", "package-name"=>"ShowCases.jl"), codeunits(body))
end

function _showcase_headers(m::Message)
    h = collect(m.headers)
    h = [startswith(η[1], ":") ? Styled(η, :blue) : Styled(η, :normal) for η ∈ h]
    h = ShowList(h, brackets="", new_lines=true, indent=repeat(" ", 8))
    h = ShowEntry(h, new_line=false, delim=Print(""))
end

function _showcase_body(m::Message)
    b = ShowLimit(Print(String(copy(m.body))), limit=53)
    b = ShowString(b, indent=repeat(" ", 8))
    b = Styled(b, :yellow)
    b = ShowEntry(b, new_line=true, indent=repeat(" ", 8), delim=Print(""))
end

function Base.show(io::IO, m::Message)
    t = ShowTypeOf(m, show_module=:never, type_style=Style(:cyan, true))
    ℓ = ShowEntry(ShowKw(length(m.body), :ℓ), new_line=false)
    b = _showcase_body(m)
    h = _showcase_headers(m)
    p = ShowList(ℓ,
                 ShowEntry(Styled(Print("Headers:"), :magenta), new_line=true, delim=Print("")),
                 h, Print(repeat(" ", 4)),
                 ShowEntry(Styled(Print("Body:"), :magenta), delim=Print("")),
                 b, Print("\n")
                 ; delim=Print(""),
                )
    show(io, t, p)
end
