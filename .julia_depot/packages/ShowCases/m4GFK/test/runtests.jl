using ShowCases
using Test

using ShowCases: withlimit, objecttype

include("utils.jl")


@testset "ShowCases.jl" begin
    @testset "limit" begin
        @testout withlimit(print, io, "spock") "spock"
        @testout withlimit(show, io, "spock") "\"spock\""
        @testout withlimit(print, io, "picard", limit=3) "pic…"
        @testout withlimit(print, io, "picard", limit=3, continuation_string="|") "pic|"
        @testout withlimit(show, io, "sisko", limit=3, check_brackets=false) "\"si…"
        @testout withlimit(show, io, "sisko", limit=3) "\"si…\""
    end

    @testset "AbstractShow" begin
        @test objecttype(Show(1)) == Int
        @test objecttype(Show(Show(1))) == Int
        @test objecttype(Int) == DataType
    end

    @testset "Show and Print" begin
        @testout show(io, Show("kirk")) "\"kirk\""
        @testout print(io, Show("kirk")) "kirk"
        @testout show(io, Print("geordi")) "geordi"
        @testout print(io, Print("geordi")) "geordi"
    end

    @testset "ShowNothing" begin
        @testout show(io, ShowNothing()) UInt8[]
        @testout print(io, ShowNothing()) UInt8[]
    end

    @testset "Style" begin
        @testout show(io, Styled("bones", color=:normal)) "\"bones\""
        @testout show(io, Styled("bones", color=:blue)) "\"bones\""
        @testout show(io, Styled("bones", bold=true)) "\"bones\""

        # ensure that it does the same thing as `printstyled`
        io1 = IOContext(IOBuffer(), :color=>true)
        io2 = IOContext(IOBuffer(), :color=>true)
        printstyled(io1, "he's dead jim", color=:blue)
        print(io2, Styled("he's dead jim", color=:blue))
        @test take!(io1.io) == take!(io2.io)

        # check case with no style
        io = IOContext(IOBuffer(), :color=>true)
        print(io, Styled("scotty"))
        @test String(take!(io.io)) == "scotty"
    end

    @testset "ShowEntry" begin
        @testout show(io, ShowEntry(1)) "1, "
        @testout show(io, ShowEntry(1, new_line=true, indent="")) "\n1, "
        @testout show(io, ShowEntry(1, new_line=true, indent="  ")) "\n  1, "
        @testout show(io, ShowEntry(1, delim=Print(":"))) "1:"
        @testout show(io, ShowEntry(1, delim=":")) "1\":\""
    end

    @testset "ShowList" begin
        l = ShowList((1,2,3))
        @test length(l) == 3
        @test l[1].object == 1
        @test l[3].object == 3
        @testout show(io, ShowList((1,2,3))) "(1, 2, 3)"
        @testout show(io, ShowList(1, 2, 3)) "(1, 2, 3)"
        @testout show(io, ShowList((1,2), brackets="[]")) "[1, 2]"
        @testout show(io, ShowList((1,2), brackets="[)")) "[1, 2)"
        @testout show(io, ShowList((1,2), left_bracket=Print("["), right_bracket=Print(")"))) "[1, 2)"
        @testout show(io, ShowList((1,2), new_lines=true)) "(\n    1, \n    2\n)"
        @testout show(io, ShowList((1,2), max=1)) "(1, …)"
    end

    @testset "types" begin
        @testout show(io, ShowType(Float64)) "Float64"
        @testout show(io, ShowType(Float64, show_module=:never)) "Float64"
        @testout show(io, ShowType(Float64, show_module=:always)) "Core.Float64"
        @testout show(io, ShowTypeOfBase(1.0)) "Float64"
        @testout show(io, ShowParams([1.0,2.0])) "{Float64, 1}"
        @testout show(io, ShowParams((1.0, 2.0), max=1)) "{Float64, …}"
        @testout show(io, ShowTypeOf(("a", 1.0))) "Tuple{String, Float64}"
        @testout show(io, ShowTypeOf(("a", 1.0), max=0)) "Tuple{…}"
    end

    @testset "misc" begin
        @testout show(io, ShowIndents(Print("a\nb\nc"), "_")) "_a\n_b\n_c"
        @testout show(io, ShowIndents(Print("a\nb"))) "    a\n    b"
        @testout show(io, ShowIndents(1)) "1"
        @testout show(io, ShowString("uhura")) "\"uhura\""
        @testout show(io, ShowString("a\nb\n")) "\"\"\"\na\nb\n\"\"\""
        @testout show(io, ShowLimit(Print("worf"), limit=2)) "wo…"
        @testout show(io, ShowKw(1, :a)) "a=1"
        @testout show(io, ShowProps((a=1, b=2))) "(a=1, b=2)"
        @testout show(io, ShowProps((a=1, b=2), show_keywords=false)) "(1, 2)"
    end

    @testset "examples" begin
        include("examples/basic.jl")
        @testout show1(io, basic_example()) "BasicExample{ComplexF64}(\n    x = 0, \n    y = 1.0, \n    z = ComplexF64[1.0 + 0.0im, -0.0 - 1.0im], \n    s = \"example\"\n)"
        @testout show2(io, basic_example()) "Main.BasicExample{Base.ComplexF64}(y=1.0, s=\"example\")"
        @testout show3(io, basic_example()) "BasicExample(0)"
        @testout show4(io, basic_example()) "BasicExample{ComplexF64}(\n    x: 0, \n    1.0\n)"

        include("examples/dirac.jl")
        @testout show(io, dirac_example()) "(0.7071067811865475 + 0.0im)|0⟩ + (0.7071067811865475 + 0.0im)|1⟩"

        include("examples/message.jl")
        @testout show(io, message_example()) "Message(ℓ=53, \n    Headers:\n        \":method\" => \"GET\", \n        \"package-name\" => \"ShowCases.jl\", \n        \":path\" => \"/\"\n    Body:\n        \"\"\"\n        This is a demonstration\n        of the\n        ShowCases.jl\n        package.\n        \"\"\"\n)"
    end
end
