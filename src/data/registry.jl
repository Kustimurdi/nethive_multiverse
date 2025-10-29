"""
Dataset Registry Module

This module provides a centralized registry for managing multiple datasets
with automatic padding/embedding for universal neural network architectures.
"""

using MLDatasets
using Flux
using Random

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

function register_svhn2(target_input_dim::Int=3072, target_output_dim::Int=10)
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
        :svhn2,
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
