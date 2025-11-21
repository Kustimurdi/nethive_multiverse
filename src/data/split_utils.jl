# Utility helpers for dataset splitting, padding and label handling
using Random
using Flux
using JLD2

"""
Normalize feature array to shape (n_features, n_samples).
Heuristics:
- If rows > cols, assume rows are samples and transpose.
- Convert element type to Float32.
Returns (X_cols, n_features, n_samples)
"""
function normalize_features(X)
    Xf = Array(X)
    if size(Xf, 1) > size(Xf, 2)
        Xc = permutedims(Xf)  # samples x features -> features x samples
    else
        Xc = Xf
    end
    # ensure Float32
    Xc = Float32.(Xc)
    return Xc, size(Xc, 1), size(Xc, 2)
end

"""
Convert labels or onehot matrix into (y_indices, y_onehot, classes).
- `labels` may be a 1D vector of labels (strings/ints)
- `onehot_labels` may be a matrix with shape (n_classes, n_samples) or (n_samples, n_classes)
`n_samples` is used to validate shapes.
"""
function labels_to_indices_and_onehot(labels, onehot_labels, n_samples)
    y = nothing
    y_onehot = nothing
    classes = nothing

    if labels !== nothing
        y = collect(labels)
        if length(y) != n_samples
            # fallback to onehot_labels if provided
            if onehot_labels !== nothing
                y_onehot = Array{Float32}(onehot_labels)
            else
                error("labels length does not match features; provide `labels` or compatible `onehot_labels`")
            end
        end
    end

    if y === nothing && y_onehot === nothing && onehot_labels !== nothing
        # attempt to coerce onehot_labels
        yoh = Array(onehot_labels)
        if size(yoh, 1) == n_samples
            yoh = permutedims(yoh)
        end
        if size(yoh, 2) != n_samples && size(yoh, 1) == n_samples
            yoh = permutedims(yoh)
        end
        if size(yoh, 2) != n_samples
            error("onehot_labels shape does not match features samples")
        end
        # ensure shape is (n_classes, n_samples)
        if size(yoh, 1) < size(yoh, 2) || size(yoh,1) == size(yoh,2)
            # OK, proceed
        end
        y_onehot = Float32.(yoh)
    end

    if y === nothing && y_onehot !== nothing
        # convert onehot to label indices (argmax per column)
        n_classes = size(y_onehot, 1)
        y = [findfirst(!iszero, view(y_onehot, :, i)) for i in 1:size(y_onehot, 2)]
    end

    if y === nothing
        error("Could not determine labels from provided data")
    end

    classes = unique(y)
    class_to_index = Dict(c => i for (i, c) in enumerate(classes))
    y_indices = [class_to_index[v] for v in y]
    if y_onehot === nothing
        y_onehot = Flux.onehotbatch(y_indices, 1:length(classes))
    else
        # ensure onehot matches ordering given by classes
        # build onehot from indices to guarantee consistency
        y_onehot = Flux.onehotbatch(y_indices, 1:length(classes))
    end

    return y_indices, y_onehot, classes
end

"""
Stratified split indices given a vector of labels (y or indices).
Returns (train_idx, test_idx) with 1-based indices into samples.
"""
function stratified_split_indices(y_indices::AbstractVector{T}, p_train::Real=0.8, seed::Integer=42) where T
    rng = Random.MersenneTwister(seed)
    label_to_inds = Dict{T, Vector{Int}}()
    for (i, lab) in enumerate(y_indices)
        push!(get!(label_to_inds, lab, Int[]), i)
    end
    train_idx = Int[]
    test_idx = Int[]
    for (lab, inds) in label_to_inds
        n = length(inds)
        order = Random.randperm(rng, n)
        ntrain = Int(floor(p_train * n))
        if ntrain < 1
            ntrain = 1
        end
        append!(train_idx, inds[order[1:ntrain]])
        if ntrain < n
            append!(test_idx, inds[order[ntrain+1:end]])
        end
    end
    train_idx = Random.shuffle(rng, train_idx)
    test_idx = Random.shuffle(rng, test_idx)
    return train_idx, test_idx
end

"""
Pad matrix rows to `target_rows`. If target_rows <= current rows, returns the original matrix (no trimming).
"""
function pad_rows(mat::AbstractMatrix, target_rows::Int; pad_value=nothing)
    current_rows = size(mat, 1)
    if target_rows <= current_rows
        return mat
    end
    pad_rows_n = target_rows - current_rows
    el = eltype(mat)
    if pad_value === nothing
        pad_value_el = zero(el)
    else
        # try to convert provided pad_value to the matrix element type
        pad_value_el = try
            convert(el, pad_value)
        catch
            error("pad_value could not be converted to element type $(el)")
        end
    end
    pad_block = fill(pad_value_el, pad_rows_n, size(mat, 2))
    return vcat(mat, pad_block)
end
