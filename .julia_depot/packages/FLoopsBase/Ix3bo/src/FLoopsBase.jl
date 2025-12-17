module FLoopsBase

using ContextVariablesX
@contextvar EXTRA_STATE_VARIABLES::Union{Vector{Union{Symbol,Expr}},Nothing} = nothing

function with_extra_state_variables(f, variables)
    current_vars = copy(something(EXTRA_STATE_VARIABLES[], Union{Symbol,Expr}[]))
    new_vars = append!(current_vars, variables)
    with_context(f, EXTRA_STATE_VARIABLES => new_vars)
end

abstract type AbstractScratchSpace end

end
