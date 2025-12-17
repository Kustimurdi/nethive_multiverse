using ShowCases, BenchmarkTools, Debugger

using ShowCases: withlimit


struct TestStruct{𝒜}
    x::Int
    y::Float64
    s::String
end

test0() = TestStruct{ComplexF64}(1, 1.0, "a")

function test1()
    ShowList((ShowKw(1, :ℓ, new_line=true), 2, 3), new_lines=true, max_entries=2,
             entry_style=ShowStyle(color=:yellow), delim_style=ShowStyle(color=:blue))
end

function test2()
    ShowList((Styled(1, color=:red), Styled(2, color=:green), Styled(3, color=:blue)))
end
