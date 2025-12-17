
@doc """
    ShowEntry

An `AbstractShow` wrapper type for showing entries in a list.  Allows for specification of
list-context options.

## Constructors
- `ShowEntry(o; new_line=false, indent="    ", delim=Print(", "))`

## Arguments
- `o`: the object to be wrapped.
- `new_line::Bool`: whether to show (or print) the entry on a new line.
- `indent::AbstractString`: If a new line is created (i.e. with `new_line=true`), the created line
    will start with this indent.
- `delim`: An object that will be shown at the end of an entry, typically a comma.  The object passed will be
    shown as is, so for example a `", "` *will* print quotes, use `Print(", ")` to omit quotes.

## Examples
```
julia> ShowEntry(1)  # by default we just get commas
1,


julia> ShowEntry("a", new_line=true, indent="__", delim=Print(": "))

__"a":
```
""" ShowEntry


@doc """
    ShowList

An `AbstractShow` wrapper type for showing iterables.  Note that, since the `ShowCases` module is intended primarily
for creating `show` methods with output resembling Julia syntax, all iterables are shown as if they are of rank 1.
Higher rank arrays should be shown with built-in methods or tools specifically intended for handling them.

Note that `ShowList` will show any element which is a `ShowEntry` object as is.  Therefore, one can pass it
`ShowEntry` objects to override defaults only for specific entries.

## Constructors
- `ShowList(o; brackets="()", left_bracket=Print("("), right_bracket=Print(")"),
            new_lines=false, indent="    ", delim=Print(", "), max=8,
            continuation=Print("…"), entry_style=nothing)`
- `ShowList(o...; kw...)` (same keyword arguments as above)

## Arguments
- `o`: Object or objects to be wrapped.  If only one is passed, it must be an iterable.
- `brackets`: A string containing the brackets to be used (e.g. `"()", "[]", "{}", "''"`)
- `left_bracket`: The left bracket to be shown.  By default, this will be computed from `brackets`.  If an argument is
    given directly, it will be shown as is, i.e. use `Print("(")` to omit quotes.
- `right_bracket`: The right bracket to be shown.  By default, this will be computed from `brackets`.  If an argument is
    given directly, it will be shown as is, i.e. use `Print(")")` to omit quotes.
- `new_lines::Bool`: Whether to print a new line for each entry.
- `indent::AbstractString`: Indent to be used
- `delim`: Delimiter to use for each entry.  This will be printed as is, so for example, use `Print(", ")` to get commas
    with spacing.
- `max::Integer`: the maximum number of entries to be printed.
- `continuation`: object to print when `max` is hit.  Will be shown as is.

## Exmaples
```
julia> ShowList(1, 2)
(1, 2)

julia> ShowList(1, 2, new_lines=true)
(
    1,
    2
)

julia> ShowList(1, 2, max=1)
(1, …)

# pass ShowEntry objects to override defaults set by `ShowList`
julia> ShowList(ShowEntry("a", delim=Print(": "), new_line=true), 1, 2)
(
    "a": 1, 2)

```
""" ShowList


@doc """
    ShowType

An `AbstractShow` wrapper for showing a type with some relevant options, such as whether the module should be shown.

Note that it is expected that `ShowType` is only used to show `Type` objects, though for generality it does not forbid
the printing of other objects.

See `ShowTypeOf` for showing the type *of* the wrapped object.

## Constructors
- `ShowType(o; show_module=:context)`

## Arguments
- `o`: the object to be wrapped.
- `show_module::Symbol`: (*valid options:* `:always`, `:never`, `:context`) specifies whether the module name should be shown.
    If `:context`, the default behavior from `Base` will be used in which the printing of the module is contextual.
    If `:never`, the module will never be shown.  If `:always`, it will always be shown.
""" ShowType


@doc """
    ShowParams

An `AbstractShow` wrapper for showing the type parameters of the type of the wrapped object.

The underlying type is determined recursively, so if an `AbstractShow` object is wrapped, the type of the (deepest) wrapped
object will be shown.

## Constructors
- `ShowParams(o; show_module=:context, kw...)`

## Arguments
- `o`: the object to be wrapped.
- `show_module::Symbol`: whether to show the modules of the type parameters, see [`ShowType`](@ref) for available options.
- `kw`: Any other keyword arguments accepted by [`ShowList`](@ref)
""" ShowParams


@doc """
    ShowString

An `AbstractShow` wrapper for showing strings. This implements some behaviors which are particularly useful for strings,
in particular, if the string contains multiple lines, block quotes (`\"\"\"`) will be used and indents will be inserted.

## Constructors
- `ShowString(o; block::Symbol=:context, indent="")`

## Arguments
- `o`: the object to be wrapped.  Typically, but not necessarily a string.
- `block::Symbol`: (*options:* `:context`, `:always`, `:never`) whether to print the string as a block.  If `:context`, this
    will be done iff the string contains multiple lines.
- `indent::AbstractString`: indent to use on new lines.
""" ShowString


@doc """
    ShowLimit

An `AbstractShow` wrapper for showing an object with the shown string limited to a fixed number of characters.

## Constructors
- `ShowLimit(o; limit=typemax(Int), n=0, continuation_string="…", check_brackets=true)`

## Arguments
- `o`: the object to be wrapped.
- `limit`: the maximum number of characters, excluding starting delimiters and the continuation string, which can be shown.
- `n::Integer`: the position of the start of the shown string.  This is useful if showing multiple outputs with a fixed limit.
    For example, `n=1` with `limit=3` is equivalent to `limit=2`.
- `continuation_string::AbstractString`: string to be printed if the limit is reached.  Not counted toward the limit.
- `check_brackets`: whether to check for the existence of opening brackets at the beginning of the shown string. This is
    useful for showing closing brackets regardless of whether the limit is reached.

## Examples
```
julia> ShowLimit(123, limit=2)
12…

julia> ShowLimit("abc", limit=2)
"a…"

julia> ShowLimit("abc", limit=2, check_brackets=false)
"a…

julia> ShowLimit(123, limit=2, n=1)
1…
```
""" ShowLimit
