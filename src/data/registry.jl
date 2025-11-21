"""
Dataset Registry Module

This module provides a centralized registry for managing multiple datasets
with automatic padding/embedding for universal neural network architectures.
"""

using MLDatasets
using Flux
using Random
using JLD2

# Helpers for tabular datasets (normalize, stratify split, padding)
include("split_utils.jl")

# Global dataset registry
const DATASET_REGISTRY = Dict{Symbol, Any}()

"""
    TaskDataset

Represents a dataset with its metadata for multi-task learning.

# Fields
- `name::Symbol`: Dataset identifier (e.g., :mnist, :fashion_mnist)
- `input_shape::Tuple`: Original input dimensions (e.g., (28, 28) for MNIST)
- `n_classes::Int`: Number of output classes
- `train_data::Tuple`: Training data (X, y)
- `test_data::Tuple`: Test data (X, y)
- `padded_input_dim::Int`: Input dimension after padding for universal models
"""
struct TaskDataset
    name::Symbol
    input_shape::Tuple
    n_classes::Int
    train_data::Tuple
    test_data::Tuple
    padded_input_dim::Int
    padded_output_dim::Int
end

"""
    register_mnist!(target_input_dim::Int=784)

Register MNIST dataset in the global registry with optional padding.

# Arguments
- `target_input_dim::Int=784`: Target input dimension for padding (default: 28*28=784)

# Side Effects
- Adds :mnist entry to DATASET_REGISTRY
- Downloads MNIST data if not already cached

The MNIST dataset has 28×28 grayscale images (784 features) and 10 classes (0-9).
"""
function register_mnist!(target_input_dim::Int=784, target_output_dim::Int=10)
    if haskey(DATASET_REGISTRY, :mnist)
        @info "MNIST already registered, skipping"
        return DATASET_REGISTRY[:mnist]
    end
    
    @info "Registering MNIST dataset..."
    
    # Load MNIST data using new MLDatasets API
    train_data = MLDatasets.MNIST(:train)
    test_data = MLDatasets.MNIST(:test)
    
    # Convert to Float32 and flatten images
    train_x = reshape(Float32.(train_data.features), 784, :)
    test_x = reshape(Float32.(test_data.features), 784, :)
    
    # Convert labels to one-hot encoding (classes 0-9)
    train_y = Flux.onehotbatch(train_data.targets, 0:9)
    test_y = Flux.onehotbatch(test_data.targets, 0:9)
    
    # Apply padding if needed
    if target_input_dim > 784
        padding_size = target_input_dim - 784
        train_x_padded = vcat(train_x, zeros(Float32, padding_size, size(train_x, 2)))
        test_x_padded = vcat(test_x, zeros(Float32, padding_size, size(test_x, 2)))
    else
        train_x_padded = train_x
        test_x_padded = test_x
    end

    if target_output_dim > 10
        padding_size = target_output_dim - 10
        train_y_padded = vcat(train_y, zeros(Float32, padding_size, size(train_y, 2)))
        test_y_padded = vcat(test_y, zeros(Float32, padding_size, size(test_y, 2)))
    else
        train_y_padded = train_y
        test_y_padded = test_y
    end
    
    dataset = TaskDataset(
        :mnist,
        (28, 28),
        10,
        (train_x_padded, train_y_padded),
        (test_x_padded, test_y_padded),
        target_input_dim,
        target_output_dim
    )
    
    DATASET_REGISTRY[:mnist] = dataset
    @info "MNIST registered successfully" input_shape=(28, 28) n_classes=10 padded_dim=target_input_dim padded_output_dim=target_output_dim
    
    return dataset
end

"""
    register_fashion_mnist!(target_input_dim::Int=784)

Register Fashion-MNIST dataset in the global registry with optional padding.

# Arguments
- `target_input_dim::Int=784`: Target input dimension for padding

Fashion-MNIST has the same structure as MNIST: 28×28 grayscale images, 10 classes.
Classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
"""
function register_fashion_mnist!(target_input_dim::Int=784, target_output_dim::Int=10)
    if haskey(DATASET_REGISTRY, :fashion_mnist)
        @info "Fashion-MNIST already registered, skipping"
        return DATASET_REGISTRY[:fashion_mnist]
    end
    
    @info "Registering Fashion-MNIST dataset..."
    
    # Load Fashion-MNIST data
    train_data = MLDatasets.FashionMNIST(:train)
    test_data = MLDatasets.FashionMNIST(:test)
    
    # Convert to Float32 and flatten images
    train_x = reshape(Float32.(train_data.features), 784, :)
    test_x = reshape(Float32.(test_data.features), 784, :)
    
    # Convert labels to one-hot encoding (classes 0-9)
    train_y = Flux.onehotbatch(train_data.targets, 0:9)
    test_y = Flux.onehotbatch(test_data.targets, 0:9)
    
    # Apply padding if needed
    if target_input_dim > 784
        padding_size = target_input_dim - 784
        train_x_padded = vcat(train_x, zeros(Float32, padding_size, size(train_x, 2)))
        test_x_padded = vcat(test_x, zeros(Float32, padding_size, size(test_x, 2)))
    else
        train_x_padded = train_x
        test_x_padded = test_x
    end

    if target_output_dim > 10
        padding_size = target_output_dim - 10
        train_y_padded = vcat(train_y, zeros(Float32, padding_size, size(train_y, 2)))
        test_y_padded = vcat(test_y, zeros(Float32, padding_size, size(test_y, 2)))
    else
        train_y_padded = train_y
        test_y_padded = test_y
    end
    
    dataset = TaskDataset(
        :fashion_mnist,
        (28, 28),
        10,
        (train_x_padded, train_y_padded),
        (test_x_padded, test_y_padded),
        target_input_dim,
        target_output_dim
    )
    
    DATASET_REGISTRY[:fashion_mnist] = dataset
    @info "Fashion-MNIST registered successfully" input_shape=(28, 28) n_classes=10 padded_dim=target_input_dim padded_output_dim=target_output_dim
    
    return dataset
end

"""
    register_cifar10!(target_input_dim::Int=3072)

Register CIFAR-10 dataset in the global registry with optional padding.

# Arguments
- `target_input_dim::Int=3072`: Target input dimension for padding (default: 32*32*3=3072)

CIFAR-10 has 32×32 RGB images (3072 features) and 10 classes.
"""
function register_cifar10!(target_input_dim::Int=3072, target_output_dim::Int=10)
    if haskey(DATASET_REGISTRY, :cifar10)
        @info "CIFAR-10 already registered, skipping"
        return DATASET_REGISTRY[:cifar10]
    end
    
    @info "Registering CIFAR-10 dataset..."
    
    # Load CIFAR-10 data
    train_data = MLDatasets.CIFAR10(:train)
    test_data = MLDatasets.CIFAR10(:test)
    
    # Convert to Float32 and flatten images (32×32×3 = 3072)
    train_x = reshape(Float32.(train_data.features), 3072, :)
    test_x = reshape(Float32.(test_data.features), 3072, :)
    
    # Convert labels to one-hot encoding (classes 0-9)
    train_y = Flux.onehotbatch(train_data.targets, 0:9)
    test_y = Flux.onehotbatch(test_data.targets, 0:9)
    
    # Apply padding if needed
    if target_input_dim > 3072
        padding_size = target_input_dim - 3072
        train_x_padded = vcat(train_x, zeros(Float32, padding_size, size(train_x, 2)))
        test_x_padded = vcat(test_x, zeros(Float32, padding_size, size(test_x, 2)))
    else
        train_x_padded = train_x
        test_x_padded = test_x
    end
    
    if target_output_dim > 10
        padding_size = target_output_dim - 10
        train_y_padded = vcat(train_y, zeros(Float32, padding_size, size(train_y, 2)))
        test_y_padded = vcat(test_y, zeros(Float32, padding_size, size(test_y, 2)))
    else
        train_y_padded = train_y
        test_y_padded = test_y
    end

    dataset = TaskDataset(
        :cifar10,
        (32, 32, 3),
        10,
        (train_x_padded, train_y_padded),
        (test_x_padded, test_y_padded),
        target_input_dim,
        target_output_dim
    )
    
    DATASET_REGISTRY[:cifar10] = dataset
    @info "CIFAR-10 registered successfully" input_shape=(32, 32, 3) n_classes=10 padded_dim=target_input_dim
    
    return dataset
end

"""
    register_cifar100!(target_input_dim::Int=3072)

Register CIFAR-100 dataset in the global registry with optional padding.

# Arguments
- `target_input_dim::Int=3072`: Target input dimension for padding (default: 32*32*3=3072)

CIFAR-100 has 32×32 RGB images (3072 features) and 100 classes.
"""
function register_cifar100!(target_input_dim::Int=3072)
    if haskey(DATASET_REGISTRY, :cifar100)
        @info "CIFAR-100 already registered, skipping"
        return DATASET_REGISTRY[:cifar100]
    end
    
    @info "Registering CIFAR-100 dataset..."
    
    # Load CIFAR-100 data
    train_data = MLDatasets.CIFAR100(:train)
    test_data = MLDatasets.CIFAR100(:test)
    
    # Convert to Float32 and flatten images (32×32×3 = 3072)
    train_x = reshape(Float32.(train_data.features), 3072, :)
    test_x = reshape(Float32.(test_data.features), 3072, :)
    
    # Convert labels to one-hot encoding (classes 0-99)
    train_y = Flux.onehotbatch(train_data.targets, 0:99)
    test_y = Flux.onehotbatch(test_data.targets, 0:99)
    
    # Apply padding if needed
    if target_input_dim > 3072
        padding_size = target_input_dim - 3072
        train_x_padded = vcat(train_x, zeros(Float32, padding_size, size(train_x, 2)))
        test_x_padded = vcat(test_x, zeros(Float32, padding_size, size(test_x, 2)))
    else
        train_x_padded = train_x
        test_x_padded = test_x
    end
    
    if target_output_dim > 100
        padding_size = target_output_dim - 100
        train_y_padded = vcat(train_y, zeros(Float32, padding_size, size(train_y, 2)))
        test_y_padded = vcat(test_y, zeros(Float32, padding_size, size(test_y, 2)))
    else
        train_y_padded = train_y
        test_y_padded = test_y
    end

    dataset = TaskDataset(
        :cifar100,
        (32, 32, 3),
        100,
        (train_x_padded, train_y_padded),
        (test_x_padded, test_y_padded),
        target_input_dim,
        target_output_dim
    )
    
    DATASET_REGISTRY[:cifar100] = dataset
    @info "CIFAR-100 registered successfully" input_shape=(32, 32, 3) n_classes=100 padded_dim=target_input_dim padded_output_dim=target_output_dim
    
    return dataset
end

function register_svhn2!(target_input_dim::Int=3072, target_output_dim::Int=10)
    if haskey(DATASET_REGISTRY, :svhn2)
        @info "SVHN2 already registered, skipping"
        return DATASET_REGISTRY[:svhn2]
    end
    
    @info "Registering SVHN2 dataset..."
    
    # Load SVHN2 data
    train_data = MLDatasets.SVHN2(:train)
    test_data = MLDatasets.SVHN2(:test)
    
    # Convert to Float32 and flatten images (32×32×3 = 3072)
    train_x = reshape(Float32.(train_data.features), 3072, :)
    test_x = reshape(Float32.(test_data.features), 3072, :)
    
    # Convert labels: SVHN2 uses 1-9 for digits 1-9 and 10 for digit 0
    # Map to 0-9: 10→0, others stay the same
    train_targets = map(x -> x == 10 ? 0 : x, train_data.targets)
    test_targets = map(x -> x == 10 ? 0 : x, test_data.targets)
    
    # Convert labels to one-hot encoding (classes 0-9)
    train_y = Flux.onehotbatch(train_targets, 0:9)
    test_y = Flux.onehotbatch(test_targets, 0:9)
    
    # Apply padding if needed
    if target_input_dim > 3072
        padding_size = target_input_dim - 3072
        train_x_padded = vcat(train_x, zeros(Float32, padding_size, size(train_x, 2)))
        test_x_padded = vcat(test_x, zeros(Float32, padding_size, size(test_x, 2)))
    else
        train_x_padded = train_x
        test_x_padded = test_x
    end
    
    if target_output_dim > 10
        padding_size = target_output_dim - 10
        train_y_padded = vcat(train_y, zeros(Float32, padding_size, size(train_y, 2)))
        test_y_padded = vcat(test_y, zeros(Float32, padding_size, size(test_y, 2)))
    else
        train_y_padded = train_y
        test_y_padded = test_y
    end

    dataset = TaskDataset(
        :svhn2,
        (32, 32, 3),
        10,
        (train_x_padded, train_y_padded),
        (test_x_padded, test_y_padded),
        target_input_dim,
        target_output_dim
    )
    
    DATASET_REGISTRY[:cifar10] = dataset
    @info "SVHN2 registered successfully" input_shape=(32, 32, 3) n_classes=10 padded_dim=target_input_dim
    
    return dataset
end


"""
    register_bank!(target_input_dim::Int = 53, target_output_dim::Int = 2; p_train::Real = 0.8, seed::Integer = 42)

Register the Bank Marketing dataset stored as a JLD2 file on disk.

Expects a folder at `/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/bank_marketing/`
containing `bank_data.jld2` with at least `features` and `labels` keys. Optionally
`onehot_labels` may be provided.
"""
function register_bank!(target_input_dim::Int = 53, target_output_dim::Int = 2; p_train::Real = 0.8, seed::Integer = 42)
    if haskey(DATASET_REGISTRY, :bank)
        @info "Bank Marketing dataset already registered, skipping"
        return DATASET_REGISTRY[:bank]
    end

    @info "Registering Bank Marketing dataset..."

    dataset_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/bank_marketing/"
    file_path = joinpath(dataset_path, "bank_data.jld2")
    if !isfile(file_path)
        error("Expected bank_data.jld2 at: $file_path")
    end

    data = Dict{String,Any}()
    jldopen(file_path, "r") do f
        for k in keys(f)
            data[string(k)] = read(f, k)
        end
    end

    features = get(data, "features", nothing)
    labels = get(data, "labels", nothing)
    onehot_labels = get(data, "onehot_labels", nothing)

    if features === nothing || labels === nothing
        error("bank_data.jld2 must contain at least `features` and `labels` entries")
    end

    # Normalize features to (n_features, n_samples)
    X_cols, n_features, n_samples = normalize_features(features)

    # Build labels as indices and one-hot matrix (n_classes x n_samples)
    y_indices, y_onehot, classes = labels_to_indices_and_onehot(labels, onehot_labels, n_samples)

    # Stratified split indices (by original labels)
    train_idx, test_idx = stratified_split_indices(y_indices, p_train, seed)

    # Build train/test splits with columns as samples
    X_train = X_cols[:, train_idx]
    X_test  = X_cols[:, test_idx]
    y_train = y_onehot[:, train_idx]
    y_test  = y_onehot[:, test_idx]

    # Pad input feature dimension to target_input_dim (no trimming)
    X_train_padded = pad_rows(X_train, target_input_dim; pad_value=0f0)
    X_test_padded  = pad_rows(X_test,  target_input_dim; pad_value=0f0)

    # Pad output dimension (one-hot rows) to target_output_dim (no trimming)
    n_classes = length(classes)
    y_train_padded = pad_rows(y_train, target_output_dim; pad_value=0f0)
    y_test_padded  = pad_rows(y_test,  target_output_dim; pad_value=0f0)

    dataset = TaskDataset(
        :bank,
        (:tabular,),
        n_classes,
        (X_train_padded, y_train_padded),
        (X_test_padded, y_test_padded),
        target_input_dim,
        target_output_dim
    )

    DATASET_REGISTRY[:bank] = dataset
    @info "Bank Marketing dataset registered successfully" input_shape=(:tabular,) n_classes=n_classes padded_dim=target_input_dim

    return dataset
end

function register_wdbc!(target_input_dim::Int = 30, target_output_dim::Int = 2; p_train::Real = 0.8, seed::Integer = 42)
    if haskey(DATASET_REGISTRY, :wdbc)
        @info "WDBC dataset already registered, skipping"
        return DATASET_REGISTRY[:wdbc]
    end

    @info "Registering WDBC dataset..."

    dataset_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/breast_cancer/"
    file_path = joinpath(dataset_path, "wdbc.jld2")
    if !isfile(file_path)
        error("Expected wdbc.jld2 at: $file_path")
    end

    data = Dict{String,Any}()
    jldopen(file_path, "r") do f
        for k in keys(f)
            data[string(k)] = read(f, k)
        end
    end

    features = get(data, "features", nothing)
    labels = get(data, "labels", nothing)
    onehot_labels = get(data, "onehot_labels", nothing)

    if features === nothing || labels === nothing
        error("wdbc.jld2 must contain at least `features` and `labels` entries")
    end

    # Normalize features to (n_features, n_samples)
    X_cols, n_features, n_samples = normalize_features(features)

    # Build labels as indices and one-hot matrix (n_classes x n_samples)
    y_indices, y_onehot, classes = labels_to_indices_and_onehot(labels, onehot_labels, n_samples)

    # Stratified split indices (by original labels)
    train_idx, test_idx = stratified_split_indices(y_indices, p_train, seed)

    # Build train/test splits with columns as samples
    X_train = X_cols[:, train_idx]
    X_test  = X_cols[:, test_idx]
    y_train = y_onehot[:, train_idx]
    y_test  = y_onehot[:, test_idx]

    # Pad input feature dimension to target_input_dim (no trimming)
    X_train_padded = pad_rows(X_train, target_input_dim; pad_value=0f0)
    X_test_padded  = pad_rows(X_test,  target_input_dim; pad_value=0f0)

    # Pad output dimension (one-hot rows) to target_output_dim (no trimming)
    n_classes = length(classes)
    y_train_padded = pad_rows(y_train, target_output_dim; pad_value=0f0)
    y_test_padded  = pad_rows(y_test,  target_output_dim; pad_value=0f0)

    dataset = TaskDataset(
        :wdbc,
        (:tabular,),
        n_classes,
        (X_train_padded, y_train_padded),
        (X_test_padded, y_test_padded),
        target_input_dim,
        target_output_dim
    )

    DATASET_REGISTRY[:wdbc] = dataset
    @info "WDBC dataset registered successfully" input_shape=(:tabular,) n_classes=n_classes padded_dim=target_input_dim
    return dataset
end

function register_iris!(target_input_dim::Int = 4, target_output_dim::Int = 3; p_train::Real = 0.8, seed::Integer = 42)
    if haskey(DATASET_REGISTRY, :iris)
        @info "Iris dataset already registered, skipping"
        return DATASET_REGISTRY[:iris]
    end

    @info "Registering Iris dataset..."

    dataset_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/iris/"
    file_path = joinpath(dataset_path, "iris.jld2")
    if !isfile(file_path)
        error("Expected iris.jld2 at: $file_path")
    end

    data = Dict{String,Any}()
    jldopen(file_path, "r") do f
        for k in keys(f)
            data[string(k)] = read(f, k)
        end
    end

    features = get(data, "features", nothing)
    labels = get(data, "labels", nothing)
    onehot_labels = get(data, "onehot_labels", nothing)

    if features === nothing || labels === nothing
        error("iris.jld2 must contain at least `features` and `labels` entries")
    end

    # Normalize features to (n_features, n_samples)
    X_cols, n_features, n_samples = normalize_features(features)

    # Build labels as indices and one-hot matrix (n_classes x n_samples)
    y_indices, y_onehot, classes = labels_to_indices_and_onehot(labels, onehot_labels, n_samples)

    # Stratified split indices (by original labels)
    train_idx, test_idx = stratified_split_indices(y_indices, p_train, seed)

    # Build train/test splits with columns as samples
    X_train = X_cols[:, train_idx]
    X_test  = X_cols[:, test_idx]
    y_train = y_onehot[:, train_idx]
    y_test  = y_onehot[:, test_idx]

    # Pad input feature dimension to target_input_dim (no trimming)
    X_train_padded = pad_rows(X_train, target_input_dim; pad_value=0f0)
    X_test_padded  = pad_rows(X_test,  target_input_dim; pad_value=0f0)

    # Pad output dimension (one-hot rows) to target_output_dim (no trimming)
    n_classes = length(classes)
    y_train_padded = pad_rows(y_train, target_output_dim; pad_value=0f0)
    y_test_padded  = pad_rows(y_test,  target_output_dim; pad_value=0f0)

    dataset = TaskDataset(
        :iris,
        (:tabular,),
        n_classes,
        (X_train_padded, y_train_padded),
        (X_test_padded, y_test_padded),
        target_input_dim,
        target_output_dim
    )

    DATASET_REGISTRY[:iris] = dataset
    @info "Iris dataset registered successfully" input_shape=(:tabular,) n_classes=n_classes padded_dim=target_input_dim
    return dataset
end

function register_car_eval!(target_input_dim::Int = 21, target_output_dim::Int = 4; p_train::Real = 0.8, seed::Integer = 42)
   if haskey(DATASET_REGISTRY, :car_eval)
        @info "Car Evaluation dataset already registered, skipping"
        return DATASET_REGISTRY[:car_eval]
    end

    @info "Registering Car Evaluation dataset..."

    dataset_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/car_evaluation/"
    file_path = joinpath(dataset_path, "car_eval.jld2")
    if !isfile(file_path)
        error("Expected car_eval.jld2 at: $file_path")
    end

    data = Dict{String,Any}()
    jldopen(file_path, "r") do f
        for k in keys(f)
            data[string(k)] = read(f, k)
        end
    end

    features = get(data, "features", nothing)
    labels = get(data, "labels", nothing)
    onehot_labels = get(data, "onehot_labels", nothing)

    if features === nothing || labels === nothing
        error("car_eval.jld2 must contain at least `features` and `labels` entries")
    end

    # Normalize features to (n_features, n_samples)
    X_cols, n_features, n_samples = normalize_features(features)

    # Build labels as indices and one-hot matrix (n_classes x n_samples)
    y_indices, y_onehot, classes = labels_to_indices_and_onehot(labels, onehot_labels, n_samples)

    # Stratified split indices (by original labels)
    train_idx, test_idx = stratified_split_indices(y_indices, p_train, seed)

    # Build train/test splits with columns as samples
    X_train = X_cols[:, train_idx]
    X_test  = X_cols[:, test_idx]
    y_train = y_onehot[:, train_idx]
    y_test  = y_onehot[:, test_idx]

    # Pad input feature dimension to target_input_dim (no trimming)
    X_train_padded = pad_rows(X_train, target_input_dim; pad_value=0f0)
    X_test_padded  = pad_rows(X_test,  target_input_dim; pad_value=0f0)

    # Pad output dimension (one-hot rows) to target_output_dim (no trimming)
    n_classes = length(classes)
    y_train_padded = pad_rows(y_train, target_output_dim; pad_value=0f0)
    y_test_padded  = pad_rows(y_test,  target_output_dim; pad_value=0f0)

    dataset = TaskDataset(
        :car_eval,
        (:tabular,),
        n_classes,
        (X_train_padded, y_train_padded),
        (X_test_padded, y_test_padded),
        target_input_dim,
        target_output_dim
    )

    DATASET_REGISTRY[:car_eval] = dataset
    @info "Car Evaluation dataset registered successfully" input_shape=(:tabular,) n_classes=n_classes padded_dim=target_input_dim
    return dataset
end

function register_wineq_red!(target_input_dim::Int = 11, target_output_dim::Int = 6; p_train::Real = 0.8, seed::Integer = 42)
   if haskey(DATASET_REGISTRY, :wineq_red)
        @info "Wine quality red dataset already registered, skipping"
        return DATASET_REGISTRY[:wineq_red]
    end

    @info "Registering wine quality red dataset..."

    dataset_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/wine_quality/"
    file_path = joinpath(dataset_path, "wineq_red.jld2")
    if !isfile(file_path)
        error("Expected wineq_red.jld2 at: $file_path")
    end

    data = Dict{String,Any}()
    jldopen(file_path, "r") do f
        for k in keys(f)
            data[string(k)] = read(f, k)
        end
    end

    features = get(data, "features", nothing)
    labels = get(data, "labels", nothing)
    onehot_labels = get(data, "onehot_labels", nothing)

    if features === nothing || labels === nothing
        error("wineq_red.jld2 must contain at least `features` and `labels` entries")
    end

    # Normalize features to (n_features, n_samples)
    X_cols, n_features, n_samples = normalize_features(features)

    # Build labels as indices and one-hot matrix (n_classes x n_samples)
    y_indices, y_onehot, classes = labels_to_indices_and_onehot(labels, onehot_labels, n_samples)

    # Stratified split indices (by original labels)
    train_idx, test_idx = stratified_split_indices(y_indices, p_train, seed)

    # Build train/test splits with columns as samples
    X_train = X_cols[:, train_idx]
    X_test  = X_cols[:, test_idx]
    y_train = y_onehot[:, train_idx]
    y_test  = y_onehot[:, test_idx]

    # Pad input feature dimension to target_input_dim (no trimming)
    X_train_padded = pad_rows(X_train, target_input_dim; pad_value=0f0)
    X_test_padded  = pad_rows(X_test,  target_input_dim; pad_value=0f0)

    # Pad output dimension (one-hot rows) to target_output_dim (no trimming)
    n_classes = length(classes)
    y_train_padded = pad_rows(y_train, target_output_dim; pad_value=0f0)
    y_test_padded  = pad_rows(y_test,  target_output_dim; pad_value=0f0)

    dataset = TaskDataset(
        :wineq_red,
        (:tabular,),
        n_classes,
        (X_train_padded, y_train_padded),
        (X_test_padded, y_test_padded),
        target_input_dim,
        target_output_dim
    )

    DATASET_REGISTRY[:wineq_red] = dataset
    @info "Wine quality red dataset registered successfully" input_shape=(:tabular,) n_classes=n_classes padded_dim=target_input_dim
    return dataset
end

function register_wineq_white!(target_input_dim::Int = 11, target_output_dim::Int = 7; p_train::Real = 0.8, seed::Integer = 42)
   if haskey(DATASET_REGISTRY, :wineq_white)
        @info "Wine quality white dataset already registered, skipping"
        return DATASET_REGISTRY[:wineq_white]
    end

    @info "Registering wine quality white dataset..."

    dataset_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/wine_quality/"
    file_path = joinpath(dataset_path, "wineq_white.jld2")
    if !isfile(file_path)
        error("Expected wineq_white.jld2 at: $file_path")
    end

    data = Dict{String,Any}()
    jldopen(file_path, "r") do f
        for k in keys(f)
            data[string(k)] = read(f, k)
        end
    end

    features = get(data, "features", nothing)
    labels = get(data, "labels", nothing)
    onehot_labels = get(data, "onehot_labels", nothing)

    if features === nothing || labels === nothing
        error("wineq_white.jld2 must contain at least `features` and `labels` entries")
    end

    # Normalize features to (n_features, n_samples)
    X_cols, n_features, n_samples = normalize_features(features)

    # Build labels as indices and one-hot matrix (n_classes x n_samples)
    y_indices, y_onehot, classes = labels_to_indices_and_onehot(labels, onehot_labels, n_samples)

    # Stratified split indices (by original labels)
    train_idx, test_idx = stratified_split_indices(y_indices, p_train, seed)

    # Build train/test splits with columns as samples
    X_train = X_cols[:, train_idx]
    X_test  = X_cols[:, test_idx]
    y_train = y_onehot[:, train_idx]
    y_test  = y_onehot[:, test_idx]

    # Pad input feature dimension to target_input_dim (no trimming)
    X_train_padded = pad_rows(X_train, target_input_dim; pad_value=0f0)
    X_test_padded  = pad_rows(X_test,  target_input_dim; pad_value=0f0)

    # Pad output dimension (one-hot rows) to target_output_dim (no trimming)
    n_classes = length(classes)
    y_train_padded = pad_rows(y_train, target_output_dim; pad_value=0f0)
    y_test_padded  = pad_rows(y_test,  target_output_dim; pad_value=0f0)

    dataset = TaskDataset(
        :wineq_white,
        (:tabular,),
        n_classes,
        (X_train_padded, y_train_padded),
        (X_test_padded, y_test_padded),
        target_input_dim,
        target_output_dim
    )

    DATASET_REGISTRY[:wineq_white] = dataset
    @info "wine quality white dataset registered successfully" input_shape=(:tabular,) n_classes=n_classes padded_dim=target_input_dim
    return dataset
end

function register_dropout!(target_input_dim::Int = 36, target_output_dim::Int = 3; p_train::Real = 0.8, seed::Integer = 42)
   if haskey(DATASET_REGISTRY, :dropout)
        @info "Student dropout dataset already registered, skipping"
        return DATASET_REGISTRY[:dropout]
    end

    @info "Registering Student Dropout dataset..."

    dataset_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/student_dropout/"
    file_path = joinpath(dataset_path, "dropout.jld2")
    if !isfile(file_path)
        error("Expected dropout.jld2 at: $file_path")
    end

    data = Dict{String,Any}()
    jldopen(file_path, "r") do f
        for k in keys(f)
            data[string(k)] = read(f, k)
        end
    end

    features = get(data, "features", nothing)
    labels = get(data, "labels", nothing)
    onehot_labels = get(data, "onehot_labels", nothing)

    if features === nothing || labels === nothing
        error("dropout.jld2 must contain at least `features` and `labels` entries")
    end

    # Normalize features to (n_features, n_samples)
    X_cols, n_features, n_samples = normalize_features(features)

    # Build labels as indices and one-hot matrix (n_classes x n_samples)
    y_indices, y_onehot, classes = labels_to_indices_and_onehot(labels, onehot_labels, n_samples)

    # Stratified split indices (by original labels)
    train_idx, test_idx = stratified_split_indices(y_indices, p_train, seed)

    # Build train/test splits with columns as samples
    X_train = X_cols[:, train_idx]
    X_test  = X_cols[:, test_idx]
    y_train = y_onehot[:, train_idx]
    y_test  = y_onehot[:, test_idx]

    # Pad input feature dimension to target_input_dim (no trimming)
    X_train_padded = pad_rows(X_train, target_input_dim; pad_value=0f0)
    X_test_padded  = pad_rows(X_test,  target_input_dim; pad_value=0f0)

    # Pad output dimension (one-hot rows) to target_output_dim (no trimming)
    n_classes = length(classes)
    y_train_padded = pad_rows(y_train, target_output_dim; pad_value=0f0)
    y_test_padded  = pad_rows(y_test,  target_output_dim; pad_value=0f0)

    dataset = TaskDataset(
        :dropout,
        (:tabular,),
        n_classes,
        (X_train_padded, y_train_padded),
        (X_test_padded, y_test_padded),
        target_input_dim,
        target_output_dim
    )

    DATASET_REGISTRY[:dropout] = dataset
    @info "Student dropout dataset registered successfully" input_shape=(:tabular,) n_classes=n_classes padded_dim=target_input_dim
    return dataset
end

function register_abalone!(target_input_dim::Int = 10, target_output_dim::Int = 28; p_train::Real = 0.8, seed::Integer = 42)
   if haskey(DATASET_REGISTRY, :abalone)
        @info "abalone dataset already registered, skipping"
        return DATASET_REGISTRY[:]
    end

    @info "Registering abalone dataset..."

    dataset_path = "/scratch/n/N.Pfaffenzeller/nikolas_nethive/datasets/abalone/"
    file_path = joinpath(dataset_path, "abalone.jld2")
    if !isfile(file_path)
        error("Expected abalone.jld2 at: $file_path")
    end

    data = Dict{String,Any}()
    jldopen(file_path, "r") do f
        for k in keys(f)
            data[string(k)] = read(f, k)
        end
    end

    features = get(data, "features", nothing)
    labels = get(data, "labels", nothing)
    onehot_labels = get(data, "onehot_labels", nothing)

    if features === nothing || labels === nothing
        error("abalone.jld2 must contain at least `features` and `labels` entries")
    end

    # Normalize features to (n_features, n_samples)
    X_cols, n_features, n_samples = normalize_features(features)

    # Build labels as indices and one-hot matrix (n_classes x n_samples)
    y_indices, y_onehot, classes = labels_to_indices_and_onehot(labels, onehot_labels, n_samples)

    # Stratified split indices (by original labels)
    train_idx, test_idx = stratified_split_indices(y_indices, p_train, seed)

    # Build train/test splits with columns as samples
    X_train = X_cols[:, train_idx]
    X_test  = X_cols[:, test_idx]
    y_train = y_onehot[:, train_idx]
    y_test  = y_onehot[:, test_idx]

    # Pad input feature dimension to target_input_dim (no trimming)
    X_train_padded = pad_rows(X_train, target_input_dim; pad_value=0f0)
    X_test_padded  = pad_rows(X_test,  target_input_dim; pad_value=0f0)

    # Pad output dimension (one-hot rows) to target_output_dim (no trimming)
    n_classes = length(classes)
    y_train_padded = pad_rows(y_train, target_output_dim; pad_value=0f0)
    y_test_padded  = pad_rows(y_test,  target_output_dim; pad_value=0f0)

    dataset = TaskDataset(
        :abalone,
        (:tabular,),
        n_classes,
        (X_train_padded, y_train_padded),
        (X_test_padded, y_test_padded),
        target_input_dim,
        target_output_dim
    )

    DATASET_REGISTRY[:abalone] = dataset
    @info "Student abalone dataset registered successfully" input_shape=(:tabular,) n_classes=n_classes padded_dim=target_input_dim
    return dataset
end


"""
    get_dataset(name::Symbol)

Retrieve a registered dataset by name.

# Arguments
- `name::Symbol`: Dataset name (e.g., :mnist, :fashion_mnist, :cifar10, :cifar100)

# Returns
- `TaskDataset`: The registered dataset

# Throws
- `KeyError`: If dataset is not registered
"""
function get_dataset(name::Symbol)
    if !haskey(DATASET_REGISTRY, name)
        throw(KeyError("Dataset $name not registered. Available: $(keys(DATASET_REGISTRY))"))
    end
    return DATASET_REGISTRY[name]
end

"""
    list_registered_datasets()

List all currently registered datasets.

# Returns
- `Vector{Symbol}`: List of registered dataset names
"""
function list_registered_datasets()
    return collect(keys(DATASET_REGISTRY))
end

"""
    clear_registry!()

Clear all registered datasets. Useful for testing.
"""
function clear_registry!()
    empty!(DATASET_REGISTRY)
    @info "Dataset registry cleared"
end

"""
    create_loaders(dataset::TaskDataset; batch_size::Int=32, shuffle_train::Bool=true)

Create Flux.DataLoader objects for training and testing.

# Arguments
- `dataset::TaskDataset`: The dataset to create loaders for
- `batch_size::Int=32`: Batch size for both train and test loaders
- `shuffle_train::Bool=true`: Whether to shuffle training data

# Returns
- `train_loader`: Flux.DataLoader for training data
- `test_loader`: Flux.DataLoader for test data

Uses the proper Flux.DataLoader instead of manual batch creation for better performance and compatibility.
"""
function create_loaders(dataset::TaskDataset; batch_size::Int=32, shuffle_train::Bool=true)
    train_x, train_y = dataset.train_data
    test_x, test_y = dataset.test_data
    
    # Create proper Flux DataLoaders
    train_loader = Flux.DataLoader(
        (train_x, train_y), 
        batchsize=batch_size, 
        shuffle=shuffle_train
    )
    
    test_loader = Flux.DataLoader(
        (test_x, test_y), 
        batchsize=batch_size, 
        shuffle=false  # Don't shuffle test data
    )
    
    return train_loader, test_loader
end
