using ShowCases


struct Bra{𝒯}
    ψ::𝒯
end

struct Ket{𝒯}
    ψ::𝒯
end

Base.show(io::IO, ψ::Bra) = show(io, ShowList((ψ.ψ,), brackets="⟨|"))
Base.show(io::IO, ψ::Ket) = show(io, ShowList((ψ.ψ,), brackets="|⟩"))


struct Linear{𝒯}
    α::ComplexF64
    a::𝒯
    β::ComplexF64
    b::𝒯
end

Base.show(io::IO, l::Linear) = show(io, ShowList((l.α,)), Show(l.a), Print(" + "), ShowList((l.β,)), Show(l.b))


function dirac_example()
    ξ = Ket(0)
    ζ = Ket(1)
    Linear(1/√(2) + 0*im, ξ, 1/√(2) + 0*im, ζ)
end
