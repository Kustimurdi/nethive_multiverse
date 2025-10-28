"""
Multi-Task Dataset Loaders Module

This module provides high-level functions for setting up multi-task datasets
and creating universal neural network models that work across different tasks.
"""

using Flux
using Statistics

include("registry.jl")

"""
    prepare_multitask_setup(dataset_names::Vector{Symbol}; batch_size::Int=32)

Prepare a complete multi-task setup with datasets, loaders, task info, and model template.

# Arguments
- `dataset_names::Vector{Symbol}`: List of datasets to include (e.g., [:mnist, :fashion_mnist])
- `batch_size::Int=32`: Batch size for all loaders

# Returns
- `loaders::Dict`: Dictionary mapping dataset names to Dict("train" => train_loader, "test" => test_loader)
- `task_info::Dict`: Dictionary mapping dataset names to TaskDataset objects
- `model_template::Function`: Function that creates fresh neural network models

This is the main entry point for multi-task setup. It handles:
1. Registering all requested datasets with proper padding
2. Creating data loaders for each dataset
3. Computing universal input/output dimensions
4. Creating a model template function

# Example
```julia
dataset_names = [:mnist, :fashion_mnist]
loaders, task_info, model_template = prepare_multitask_setup(dataset_names)

# Create a fresh model
model = model_template()

# Get training data for MNIST
train_loader = loaders[:mnist]["train"]
```
"""
function prepare_multitask_setup(dataset_names::Vector{Symbol}; batch_size::Int=32)
    @info "Preparing multi-task setup for datasets: $dataset_names"
    
    # Step 1: Calculate universal dimensions
    max_input_dim, max_output_dim = calculate_universal_dimensions(dataset_names)
    @info "Universal dimensions" max_input_dim=max_input_dim max_output_dim=max_output_dim
    
    # Step 2: Register all datasets with padding
    task_info = Dict{Symbol, TaskDataset}()
    for dataset_name in dataset_names
        dataset = register_dataset_with_padding!(dataset_name, max_input_dim, max_output_dim)
        task_info[dataset_name] = dataset
    end
    
    # Step 3: Create loaders for all datasets
    loaders = Dict{Symbol, Dict{String, Any}}()
    for dataset_name in dataset_names
        dataset = task_info[dataset_name]
        train_loader, test_loader = create_loaders(dataset, batch_size=batch_size)
        loaders[dataset_name] = Dict(
            "train" => train_loader,
            "test" => test_loader
        )
    end
    
    # Step 4: Create model template function
    model_template = create_universal_model_template(max_input_dim, max_output_dim)
    
    @info "Multi-task setup completed" n_datasets=length(dataset_names) batch_size=batch_size
    
    return loaders, task_info, model_template
end

"""
    calculate_universal_dimensions(dataset_names::Vector{Symbol})

Calculate the maximum input and output dimensions across all specified datasets.

# Arguments
- `dataset_names::Vector{Symbol}`: List of dataset names

# Returns
- `max_input_dim::Int`: Maximum input dimension (after flattening)
- `max_output_dim::Int`: Maximum number of output classes

This function determines the universal architecture size needed to accommodate all tasks.
"""
function calculate_universal_dimensions(dataset_names::Vector{Symbol})
    dataset_specs = Dict(
        :mnist => (input_dim=784, n_classes=10),          # 28*28 = 784
        :fashion_mnist => (input_dim=784, n_classes=10),  # 28*28 = 784
        :cifar10 => (input_dim=3072, n_classes=10),       # 32*32*3 = 3072
        :cifar100 => (input_dim=3072, n_classes=100),     # 32*32*3 = 3072
    )
    
    max_input_dim = 0
    max_output_dim = 0
    
    for dataset_name in dataset_names
        if !haskey(dataset_specs, dataset_name)
            throw(ArgumentError("Unknown dataset: $dataset_name. Supported: $(keys(dataset_specs))"))
        end
        
        spec = dataset_specs[dataset_name]
        max_input_dim = max(max_input_dim, spec.input_dim)
        max_output_dim = max(max_output_dim, spec.n_classes)
    end
    
    return max_input_dim, max_output_dim
end

"""
    register_dataset_with_padding!(dataset_name::Symbol, target_input_dim::Int)

Register a dataset with the specified input dimension padding.

# Arguments
- `dataset_name::Symbol`: Name of the dataset to register
- `target_input_dim::Int`: Target input dimension for padding

# Returns
- `TaskDataset`: The registered dataset

Automatically calls the appropriate registration function based on dataset name.
"""
function register_dataset_with_padding!(dataset_name::Symbol, target_input_dim::Int, target_output_dim::Int)
    if dataset_name == :mnist
        return register_mnist!(target_input_dim, target_output_dim)
    elseif dataset_name == :fashion_mnist
        return register_fashion_mnist!(target_input_dim, target_output_dim)
    elseif dataset_name == :cifar10
        return register_cifar10!(target_input_dim, target_output_dim)
    else
        throw(ArgumentError("Unknown dataset: $dataset_name"))
    end
end

"""
    create_universal_model_template(input_dim::Int, output_dim::Int)

Create a function that generates fresh neural network models for multi-task learning.

# Arguments
- `input_dim::Int`: Input dimension (padded to universal size)
- `output_dim::Int`: Output dimension (universal across all tasks)

# Returns
- `model_template::Function`: Function that creates fresh model instances

The returned function creates neural networks suitable for all tasks in the multi-task setup.
Each call to `model_template()` returns a fresh model with randomly initialized weights.

# Example
```julia
model_template = create_universal_model_template(3072, 10)
model1 = model_template()  # Fresh model
model2 = model_template()  # Another fresh model (different weights)
```
"""
function create_universal_model_template(input_dim::Int, output_dim::Int)
    function model_template()
        # Universal architecture that works for all classification tasks
        hidden_dim1 = min(128, input_dim ÷ 4)  # Adaptive hidden size
        hidden_dim2 = min(64, hidden_dim1 ÷ 2)
        
        return Chain(
            Dense(input_dim, hidden_dim1, relu),
            Dense(hidden_dim1, hidden_dim2, relu),
            Dense(hidden_dim2, output_dim)  # No activation (logits for cross-entropy)
        )
    end
    
    @info "Created universal model template" input_dim=input_dim output_dim=output_dim
    return model_template
end

"""
    test_dataset_loading(dataset_names::Vector{Symbol}=[:mnist, :fashion_mnist])

Test the dataset loading functionality with the specified datasets.

# Arguments
- `dataset_names::Vector{Symbol}`: Datasets to test (default: MNIST and Fashion-MNIST)

# Returns
- `Bool`: True if all tests pass

This function validates that datasets can be loaded, padded, and used for training.
"""
function test_dataset_loading(dataset_names::Vector{Symbol}=[:mnist, :fashion_mnist])
    @info "Testing dataset loading for: $dataset_names"
    
    try
        # Test the full setup
        loaders, task_info, model_template = prepare_multitask_setup(dataset_names, batch_size=16)
        
        # Test model creation
        model = model_template()
        @info "Model created" n_layers=length(model)
        
        # Test data shapes
        for dataset_name in dataset_names
            dataset = task_info[dataset_name]
            train_loader = loaders[dataset_name]["train"]
            
            # Get first batch
            if !isempty(train_loader)
                x_batch, y_batch = train_loader[1]
                output = model(x_batch)
                
                @info "Dataset test successful" dataset=dataset_name input_shape=size(x_batch) output_shape=size(y_batch) model_output_shape=size(output)
                
                # Validate shapes
                @assert size(x_batch, 1) == dataset.padded_input_dim "Input dimension mismatch"
                @assert size(y_batch, 1) == dataset.n_classes "Output classes mismatch"
                @assert size(output, 1) == dataset.n_classes "Model output dimension mismatch"
            end
        end
        
        @info "All dataset loading tests passed!"
        return true
        
    catch e
        @error "Dataset loading test failed" error=e
        return false
    end
end

"""
    create_universal_model(input_dim::Int, output_dim::Int; 
                          hidden_dims::Vector{Int}=[128, 64])

Create a single universal neural network model.

# Arguments
- `input_dim::Int`: Input dimension
- `output_dim::Int`: Output dimension  
- `hidden_dims::Vector{Int}=[128, 64]`: Hidden layer dimensions

# Returns
- `Flux.Chain`: Neural network model

This is a simpler interface for creating single models when you don't need a template function.
"""
function create_universal_model(input_dim::Int, output_dim::Int; 
                               hidden_dims::Vector{Int}=[128, 64])
    layers = []
    
    # Input layer
    push!(layers, Dense(input_dim, hidden_dims[1], relu))
    
    # Hidden layers
    for i in 2:length(hidden_dims)
        push!(layers, Dense(hidden_dims[i-1], hidden_dims[i], relu))
    end
    
    # Output layer (no activation for logits)
    push!(layers, Dense(hidden_dims[end], output_dim))
    
    return Chain(layers...)
end
