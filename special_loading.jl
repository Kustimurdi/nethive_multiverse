using Random
using JLD2
using Flux

if load_info !== nothing
    mapping_path = joinpath(output_dir, foldername, "loaded_bee_mapping.jdl2")
    mkpath(dirname(mapping_path))
    JLD2.@save mapping_path load_info
end

function load_models_into_hive!(hive::MultiTaskHive, model_dir::String)
    """Load pre-trained models into the hive from specified directory"""
    for (bee_idx, brain) in enumerate(hive.brains)
        model_path = joinpath(model_dir, "bee_$(bee_idx)_model.jdl2")
        if isfile(model_path)
            @info "Loading model for bee $(bee_idx) from $model_path"
            JLD2.@load model_path model_state
            Flux.loadmodel!(brain, model_state)
        else
            @warn "Model file not found for bee $(bee_idx): $model_path"
        end
    end
    JLD2.@load joinpath(model_dir, "suppressed_tasks.jdl2") suppressed_tasks suppression_time_left
    overwrite_overlap!(hive.suppressed_tasks, suppressed_tasks)
    overwrite_overlap!(hive.suppression_start_times, suppression_time_left)
    println("models have been loaded")
    println("quick test")
    println("are the first elements the same?: ")
    println(hive.suppression_start_times[1,1] == suppression_time_left[1,1])
    println(hive.suppressed_tasks[1,1] == suppressed_tasks[1,1])
    return nothing
end

"""
Overwrite the overlapping top-left region of `dest` with `src`.
Only the overlapping entries are overwritten; the rest of `dest` is left untouched.
This works in-place and returns `dest`.
"""
function overwrite_overlap!(dest::AbstractMatrix, src::AbstractMatrix)
    m = min(size(dest,1), size(src,1))
    n = min(size(dest,2), size(src,2))
    dest[1:m, 1:n] .= src[1:m, 1:n]
    return dest
end
