
"""
    @testout(𝒻, expr)

Check that calling `𝒻` on an `IOBuffer` `io` fills the buffer with data equal to `expr`.

If `expr` is a string, it will be compared to `String(take!(io))`, else `take!(io)` (this method is better
than taking `codeunits` because test failure messages will show strings).
"""
macro testout(𝒻, expr)
    quote
        io = IOBuffer()
        $𝒻
        @test $expr == ($expr isa AbstractString ? String : identity)(take!(io))
    end
end

function make_string(𝒻)
    io = IOBuffer()
    𝒻(io)
    String(take!(io))
end
