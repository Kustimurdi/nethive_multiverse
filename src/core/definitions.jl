"""
Multi-task Neural Bee System - Core Definitions

This module defines the data structures for multi-task hive simulations where
bees (neural networks) can work on multiple classification tasks simultaneously.
"""


"""
    MultiTaskHiveConfig

Configuration for multi-task hive simulation where bees work on multiple classification datasets.

# Fields
## Multi-task Setup
- `dataset_names::Vector{Symbol}`: List of datasets for the experiment
- `n_tasks::Int`: Number of tasks (length of dataset_names)
- `max_input_dim::Int`: Maximum input dimension across all tasks  
- `max_output_dim::Int`: Maximum output dimension across all tasks
- `model_template::Function`: Function that creates fresh neural network models
- `task_mapping::Dict{Symbol, Int}`: Maps dataset names to task indices

## Simulation Parameters
- `n_bees::UInt16`: Number of bees in the hive
- `n_epochs::UInt64`: Number of simulation epochs
- `n_steps_per_epoch::UInt16`: Gillespie steps per epoch
- `production_rate::Float64`: Rate at which bees improve their task performance
- `interaction_rate::Float64`: Rate of bee-to-bee interactions
- `learning_rate::Float32`: Learning rate for neural network training
- `punish_rate::Float32`: Rate for interaction punishment effects
- `lambda_sensitivity::Float16`: Interaction sensitivity parameter
- `random_seed::Int`: Random seed for reproducibility
- `save_nn_epochs::Int`: Frequency for saving neural network states

## Legacy Compatibility
- `dataset_name::String`: Primary dataset name
- `parent_dataset_name::String`: Parent dataset identifier  
- `task_config`: Task-specific configuration

Note: This configuration assumes classification tasks and uses accuracy directly as the queen gene proxy.
"""
struct MultiTaskHiveConfig
    # Multi-task fields
    dataset_names::Vector{Symbol}
    n_tasks::Int
    max_input_dim::Int
    max_output_dim::Int
    model_template::Function
    task_to_index_mapping::Dict{Symbol, Int}
    index_to_task_mapping::Dict{Int, Symbol}
    
    # Simulation parameters
    n_bees::Int
    n_epochs::Int
    n_steps_per_epoch::Int
    production_rate::Float64
    interaction_rate::Float64
    learning_rate::Float64
    punish_rate::Float64
    lambda_sensitivity::Float64
    punishment::Symbol
    random_seed::Int
    save_nn_epochs::Int
    batches_per_step::Union{Int, Nothing}
    dead_time::Float64
    
    function MultiTaskHiveConfig(dataset_names::Vector{Symbol},
                                model_template::Function,
                                max_input_dim::Int,
                                max_output_dim::Int,
                                n_bees::Int,
                                n_epochs::Int,
                                n_steps_per_epoch::Int,
                                production_rate::Float64,
                                interaction_rate::Float64,
                                learning_rate::Float64,
                                punish_rate::Float64,
                                lambda_sensitivity::Float64,
                                punishment::Symbol,
                                random_seed::Int,
                                save_nn_epochs::Int,
                                batches_per_step::Union{Int, Nothing},
                                dead_time::Float64)
        
        # Validate parameters
        if n_bees < 1
            throw(ArgumentError("Number of bees must be at least 1"))
        end
        if n_epochs < 1
            throw(ArgumentError("Number of epochs must be at least 1"))
        end
        if learning_rate <= 0
            throw(ArgumentError("Learning rate must be positive"))
        end
        if punish_rate <= 0
            throw(ArgumentError("Punish rate must be positive"))
        end

        if production_rate <= 0 || interaction_rate < 0
            throw(ArgumentError("Production and interaction rates must be non-negative"))
        end
        # Validate punishment option
        valid_punishments = (:resetting, :time_out, :none, :gradient_ascend)
        if !(punishment in valid_punishments)
            throw(ArgumentError("Invalid punishment type: $punishment. Valid options: $(valid_punishments)"))
        end

        if lambda_sensitivity < 0
            throw(ArgumentError("Lambda sensitivity must be non-negative"))
        end
        if length(dataset_names) == 0
            throw(ArgumentError("At least one dataset must be specified"))
        end
        if max_input_dim <= 0 || max_output_dim <= 0
            throw(ArgumentError("Model dimensions must be positive"))
        end
        
        # Create task mapping
        n_tasks = length(dataset_names)
        task_to_index_mapping = Dict{Symbol, Int}()
        index_to_task_mapping = Dict{Int, Symbol}()
        for (i, dataset_name) in enumerate(dataset_names)
            task_to_index_mapping[dataset_name] = i
            index_to_task_mapping[i] = dataset_name
        end
        
        return new(dataset_names,
                n_tasks,
                max_input_dim,
                max_output_dim,
                model_template,
                task_to_index_mapping,
                index_to_task_mapping,
                n_bees,
                n_epochs,
                n_steps_per_epoch,
                production_rate,
                interaction_rate,
                learning_rate,
                punish_rate,
                lambda_sensitivity,
                punishment,
                random_seed,
                save_nn_epochs,
                batches_per_step,
                dead_time)
    end
end

"""
    create_multitask_hive_config(dataset_names, loaders, task_info, model_template; kwargs...)

Create a MultiTaskHiveConfig for multi-task classification experiments using output from prepare_multitask_setup().

# Arguments
- `dataset_names::Vector{Symbol}`: List of dataset names
- `loaders`: Dict from prepare_multitask_setup (used for validation)
- `task_info`: Dict from prepare_multitask_setup  
- `model_template::Function`: Function that creates fresh neural network models
- Additional keyword arguments for simulation parameters

# Returns  
- `MultiTaskHiveConfig`: Configured for multi-task classification experiment

Note: This function assumes classification tasks and uses accuracy as the queen gene proxy.

# Example
```julia
dataset_names = [:mnist, :fashion_mnist]
loaders, task_info, model_template = prepare_multitask_setup(dataset_names)
config = create_multitask_hive_config(dataset_names, loaders, task_info, model_template)
```
"""
function create_multitask_hive_config(dataset_names::Vector{Symbol}, 
                                    loaders::Dict, 
                                    task_info::Dict,
                                    model_template::Function;
                                    n_bees::Int=5,
                                    n_epochs::Int=10,
                                    n_steps_per_epoch::Int=100,
                                    production_rate::Float64=1.0,
                                    interaction_rate::Float64=0.5,
                                    learning_rate::Float64=0.001,
                                    punish_rate::Float64=0.01,
                                    punishment::Symbol = :gradient_ascend,
                                    lambda_sensitivity::Float64=1.0,
                                    random_seed::Int=42,
                                    save_nn_epochs::Int=10,
                                    batches_per_step::Union{Int, Nothing}=nothing,
                                    dead_time::Float64=1.0)
    
    # Calculate max dimensions from task_info
    max_input_dim = 0
    max_output_dim = 0
    for (name, dataset) in task_info
        input_dim = prod(dataset.input_shape)
        max_input_dim = max(max_input_dim, input_dim)
        max_output_dim = max(max_output_dim, dataset.n_classes)
    end
    # Validate that loaders exist for all datasets and are classification tasks
    for dataset_name in dataset_names
        if !haskey(loaders, dataset_name)
            throw(ArgumentError("Missing loader for dataset: $dataset_name"))
        end
        
        # Validate that all tasks are classification (have discrete classes)
        if !haskey(task_info, dataset_name)
            throw(ArgumentError("Missing task info for dataset: $dataset_name"))
        end
        
        if task_info[dataset_name].n_classes <= 1
            throw(ArgumentError("Dataset $dataset_name does not appear to be a classification task (n_classes = $(task_info[dataset_name].n_classes))"))
        end
    end
    
    @info "Creating multi-task classification hive config" datasets=dataset_names n_tasks=length(dataset_names) max_input_dim=max_input_dim max_output_dim=max_output_dim
    
    return MultiTaskHiveConfig(
        dataset_names,
        model_template,
        max_input_dim,
        max_output_dim,
        UInt16(n_bees),
        UInt64(n_epochs),
        UInt16(n_steps_per_epoch),
        production_rate,
        interaction_rate,
        Float32(learning_rate),
        Float32(punish_rate),
        punishment,
        Float16(lambda_sensitivity),
        random_seed,
        save_nn_epochs,
        batches_per_step,
        dead_time
    )
end

"""
    MultiTaskHive

Simplified hive-centric representation for multi-task bee colony simulations.
All bees have identical neural networks and train on tasks with equal probability.

# Fields
## Configuration
- `config::MultiTaskHiveConfig`: Configuration for this hive
- `n_bees::Int`: Number of bees in the hive
- `n_tasks::Int`: Number of tasks

## Simulation State
- `current_epoch::UInt32`: Current simulation epoch
- `current_time::Float64`: Current simulation time

## Neural Networks (Parallel Arrays)
- `brains::Vector{Flux.Chain}`: Neural network models for each bee

## Performance State (Matrices: n_bees × n_tasks)
- `queen_genes::Matrix{Float64}`: Performance matrix [bee, task] in [0,1]

## Interaction State (Matrices: n_bees × n_tasks)  
- `suppressed_tasks::Matrix{Bool}`: Suppression status [bee, task]
- `suppression_start_times::Matrix{Float64}`: When suppression started [bee, task]

This design enables efficient matrix operations for Gillespie algorithms with equal task probability.
All bees train on tasks with uniform probability determined by the production rate.
"""
mutable struct MultiTaskHive
    # Configuration
    config::MultiTaskHiveConfig
    n_bees::Int
    n_tasks::Int
    
    current_epoch::UInt32
    current_time::Float64
    
    # Neural networks (parallel arrays)
    brains::Vector{Flux.Chain}
    
    # Performance state (n_bees × n_tasks matrices)
    queen_genes::Matrix{Float64}              # [bee, task] performance in [0,1]
    losses::Matrix{Float64}
    
    # Interaction state (n_bees × n_tasks matrices)
    suppressed_tasks::Matrix{Bool}            # [bee, task] suppression status
    suppression_start_times::Matrix{Float64}  # [bee, task] suppression timing

    function MultiTaskHive(config::MultiTaskHiveConfig;
                          initial_queen_genes::Union{Matrix{Float64}, Nothing}=nothing)
        
        n_bees = Int(config.n_bees)
        n_tasks = config.n_tasks
        
        # Validate dimensions
        if n_bees <= 0 || n_tasks <= 0
            throw(ArgumentError("Number of bees and tasks must be positive"))
        end
        
        # Create neural networks
        brains = [config.model_template() for _ in 1:n_bees]
        
        # Initialize queen genes (performance matrix)
        if initial_queen_genes === nothing
            queen_genes = zeros(Float64, n_bees, n_tasks)
        else
            if size(initial_queen_genes) != (n_bees, n_tasks)
                throw(ArgumentError("initial_queen_genes must be $n_bees × $n_tasks matrix"))
            end
            queen_genes = clamp.(initial_queen_genes, 0.0, 1.0)
        end

        losses = zeros(Float64, n_bees, n_tasks)
        
        # Initialize simulation state
        current_epoch = UInt32(0)
        current_time = 0.0
        
        # Initialize interaction state  
        suppressed_tasks = fill(false, n_bees, n_tasks)
        suppression_start_times = zeros(Float64, n_bees, n_tasks)
        
        return new(config,
                  n_bees,
                  n_tasks,
                  current_epoch,
                  current_time,
                  brains,
                  losses,
                  queen_genes,
                  suppressed_tasks,
                  suppression_start_times)
    end
end

"""
    create_multitask_hive(config::MultiTaskHiveConfig; kwargs...)

Create a MultiTaskHive from a MultiTaskHiveConfig.

# Arguments
- `config::MultiTaskHiveConfig`: Configuration for the hive
- Optional keyword arguments passed to MultiTaskHive constructor

# Returns
- `MultiTaskHive`: Initialized hive with fresh neural networks

# Example
```julia
config = create_multitask_hive_config([:mnist, :fashion_mnist], loaders, task_info, model_template)
hive = create_multitask_hive(config)
```
"""
function create_multitask_hive(config::MultiTaskHiveConfig; kwargs...)
    return MultiTaskHive(config; kwargs...)
end

mutable struct GillespieEventLog
    time::Vector{Float64}
    bee1_id::Vector{Int}
    bee2_id::Vector{Int}
    task_id::Vector{Int}
    accuracies::Vector{Vector{Float64}}  # or Matrix{Float64} if task count is fixed
end

function GillespieEventLog()
    GillespieEventLog(Float64[], Int[], Int[], Int[], Vector{Vector{Float64}}())
end

function push_event!(log::GillespieEventLog, time, bee1, bee2, task, accuracies)
    push!(log.time, time)
    push!(log.bee1_id, bee1)
    push!(log.bee2_id, bee2)
    push!(log.task_id, task)
    push!(log.accuracies, accuracies)
end
