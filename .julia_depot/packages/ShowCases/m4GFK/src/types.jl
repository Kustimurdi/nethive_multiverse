
struct ShowType{𝒯} <: AbstractShow{𝒯}
    object::𝒯
    mod::Symbol
end

ShowType(o; show_module::Symbol=:context) = ShowType(o, show_module)

function Base.show(io::IO, s::ShowType)
    str = first(split(string(s.object), "{"))
    if s.mod == :always
        occursin(".", str) || (str = string(parentmodule(s.object))*"."*str)
    elseif s.mod == :never
        str = last(split(str, "."))
    end
    print(io, str)
end


"""
    ShowTypeOfBase(o; kw...)

Use `ShowType` on the type of the object `o`, passing keyword arguments to that constructor.  See [`ShowType`](@ref).
"""
ShowTypeOfBase(o; kw...) = ShowType(objecttype(o); kw...)


function typeparams(o)
    𝒯 = objecttype(o)
    𝒯 isa DataType ? 𝒯.parameters : DataType[]
end

function ShowParams(o; show_module::Symbol=:context, brackets::AbstractString="{}", kw...)
    p = typeparams(o)
    p = ntuple(i -> ShowType(p[i]; show_module), length(p))
    isempty(p) ? ShowNothing() : ShowList(p; brackets, kw...)
end

"""
    ShowTypeOf(o, type_style=Style(), show_module=:context, show_params=true, kw...)

Show the type of object `o`.

## Arguments
- `type_style::Style`: Style to use for the type.
- `show_module::Symbol`: whether to show modules for the type and its parameters.  See [`ShowType`](@ref) for options.
- `kw`: All other keyword arguments passed to [`ShowParams`](@ref)
"""
function ShowTypeOf(o; type_style::Style=Style(), show_module::Symbol=:context, show_params::Bool=true, kw...)
    ShowCat(Styled(ShowTypeOfBase(o; show_module), type_style),
            show_params ? ShowParams(o; show_module, kw...) : ShowNothing())
end
