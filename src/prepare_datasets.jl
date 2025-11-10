function prepare_bank_marketing(directory_path::String)
    df = CSV.read(joinpath(directory_path, "bank.csv"), DataFrame)
    df_names = names(df)
    new_df = DataFrame()
    for name in df_names
        println("Processing column: $name")
        println("Type: ", unique(typeof.(df[!, name]))[1])
        
        types = unique(typeof.(df[!, name]))
        println("Unique types in column: ", types)
    end
    return df
end

df = prepare_bank_marketing("/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/bank_marketing/")

function preprocess_dataset(df::DataFrame; label_col::Union{Symbol, Int, Nothing}=nothing, pad_value::Float32=0f0)
    """
    Preprocess a DataFrame into Float32 feature vectors.

    - One-hot encodes categorical/string columns.
    - Converts numeric columns to Float32.
    - Flattens per-row features into a single vector and pads rows to uniform length.

    Returns (X, y, encoders) where X is an (n_samples x n_features) Float32 matrix,
    y is an empty Vector (label handling is optional), and encoders is a Dict mapping
    column Symbols to their value->index dictionaries for categorical columns.
    """
    feature_vectors = Vector{Vector{Float32}}()
    encoders = Dict{Symbol, Dict{Any, Int}}()
    n_cat = Dict{Symbol, Int}()

    cols = names(df)
    # normalize label_col to Symbol if given as Int
    if isa(label_col, Int)
        label_col = cols[label_col]
    end

    # detect categorical columns and build encoders
    for col in cols
        if label_col !== nothing && col == label_col
            continue
        end
        col_sym = Symbol(col)
        col_vals = df[!, col]
        # treat as categorical if values are strings or categorical types
        is_cat = all(x -> ismissing(x) || x isa AbstractString, col_vals)
        if !is_cat
            t = eltype(col_vals)
            # CategoricalValue may come from CategoricalArrays.jl
            is_cat = t <: CategoricalValue || t <: Union{Missing, CategoricalValue}
        end

        if is_cat
            unique_vals = collect(skipmissing(unique(col_vals)))
            encoders[col_sym] = Dict(v => i for (i, v) in enumerate(unique_vals))
            n_cat[col_sym] = length(unique_vals)
        end
    end

    # build feature vectors
    for row in eachrow(df)
        feats = Float32[]
        for col in cols
            if label_col !== nothing && col == label_col
                continue
            end
            col_sym = Symbol(col)
            if haskey(encoders, col_sym)
                encoder = encoders[col_sym]
                m = n_cat[col_sym]
                onehot = zeros(Float32, m)
                v = row[col]
                if !ismissing(v) && haskey(encoder, v)
                    onehot[encoder[v]] = 1f0
                end
                append!(feats, onehot)
            else
                v = row[col]
                if ismissing(v)
                    push!(feats, pad_value)
                elseif v isa Number
                    push!(feats, Float32(v))
                else
                    # try converting other scalars to Float32, otherwise pad
                    try
                        push!(feats, Float32(v))
                    catch
                        push!(feats, pad_value)
                    end
                end
            end
        end
        push!(feature_vectors, feats)
    end

    # pad to uniform length (if necessary)
    if isempty(feature_vectors)
        X = zeros(Float32, 0, 0)
    else
        max_len = maximum(length.(feature_vectors))
        for i in eachindex(feature_vectors)
            len = length(feature_vectors[i])
            if len < max_len
                feature_vectors[i] = vcat(feature_vectors[i], fill(pad_value, max_len - len))
            end
        end
        # hcat the vectors (produces D x N), then transpose to N x D
        X = permutedims(hcat(feature_vectors...))
    end

    y = Vector{Any}() # label handling left to caller (optional)
    return X, y, encoders
end

function encode_labels(labels::AbstractVector; classes::Union{AbstractVector, Nothing}=nothing, return_onehot::Bool=true)
    """
    Encode a vector of labels into integer indices and optionally a one-hot Float32 matrix.

    Arguments
    - labels: vector of label values (can be strings, ints, symbols, ...)
    - classes: optional vector specifying the class ordering to use. If `nothing`, the
      unique values in `labels` (in order of first appearance) are used.
    - return_onehot: if true, also return a (n_samples x n_classes) Float32 one-hot matrix.

    Returns
    - y_idx::Vector{Int}: 1-based class indices for each label
    - y_onehot::Matrix{Float32} (optional): one-hot encoded matrix (n_samples, n_classes)
    - classes::Vector: the ordered list of classes used

    Note: missing values in `labels` are not allowed and will raise an error.
    """
    if any(ismissing, labels)
        throw(ArgumentError("encode_labels does not accept missing values in labels"))
    end

    if classes === nothing
        # preserve order of first appearance
        seen = Dict{Any,Bool}()
        cls = Any[]
        for v in labels
            if !haskey(seen, v)
                push!(cls, v)
                seen[v] = true
            end
        end
        classes = cls
    end

    class_to_idx = Dict{Any,Int}(c => i for (i,c) in enumerate(classes))
    n = length(labels)
    k = length(classes)
    y_idx = Vector{Int}(undef, n)
    for (i, v) in enumerate(labels)
        if !haskey(class_to_idx, v)
            throw(ArgumentError("Label value $(v) not found in provided classes"))
        end
        y_idx[i] = class_to_idx[v]
    end

    if return_onehot
        Y = zeros(Float32, n, k)
        for i in 1:n
            Y[i, y_idx[i]] = 1f0
        end
        return y_idx, Y, classes
    else
        return y_idx, classes
    end
end

directory_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/bank_marketing"
bank_df = CSV.read(joinpath(directory_path, "bank.csv"), DataFrame)
bank_full_df = CSV.read(joinpath(directory_path, "bank-full.csv"), DataFrame)

x, y, enc = preprocess_dataset(bank_df, label_col=:y)
x_full, y_full, enc_full = preprocess_dataset(bank_full_df, label_col=:y)

y, classes = encode_labels(bank_df.y)

using MLDatasets
MLDatasets.MNIST