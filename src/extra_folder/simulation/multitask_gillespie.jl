"""
Multi-Task Gillespie Simulation Module

This module implements the Gillespie stochastic simulation algorithm adapted for multi-task
neural bee colonies where bees can train on multiple tasks with equal probability.
"""

using Random
using Distributions
using Statistics
using Logging

# Include our training functions
include("../training/multitask_training.jl")

"""
    MultiTaskEvent

Represents a stochastic event in the multi-task Gillespie simulation.

# Fields
- `event_type::Symbol`: Type of event (:train, :interact, :decay)
- `bee_idx::Int`: Index of the bee involved
- `task_idx::Int`: Index of the task involved  
- `partner_bee_idx::Union{Int,Nothing}`: Partner bee for interactions (nothing for train/decay)
- `timestamp::Float64`: Simulation time when event occurred
"""
struct MultiTaskEvent
    event_type::Symbol  # :train, :interact, :decay
    bee_idx::Int
    task_idx::Int
    partner_bee_idx::Union{Int,Nothing}
    timestamp::Float64
end

"""
    calculate_multitask_propensities(hive::MultiTaskHive, config::MultiTaskHiveConfig)

Calculate the propensity (rate) for each possible event in the multi-task system.

# Arguments
- `hive::MultiTaskHive`: Current hive state
- `config::MultiTaskHiveConfig`: Configuration parameters

# Returns
- `train_propensities::Matrix{Float64}`: Training rates [bee, task] 
- `interact_propensities::Matrix{Float64}`: Interaction rates [bee, task]
- `total_propensity::Float64`: Sum of all propensities

For simplified equal-probability system:
- All bees have equal training propensity on all tasks
- Interaction rates depend on performance differences (K-matrix style)
"""
function calculate_multitask_propensities(hive::MultiTaskHive, config::MultiTaskHiveConfig)
    n_bees = hive.n_bees
    n_tasks = hive.n_tasks
    
    # Training propensities: equal for all [bee, task] combinations
    base_train_rate = config.production_rate / (n_bees * n_tasks)
    train_propensities = fill(base_train_rate, n_bees, n_tasks)
    
    # Interaction propensities: based on performance differences
    interact_propensities = zeros(Float64, n_bees, n_tasks)
    
    for task_idx in 1:n_tasks
        # Get performance vector for this task
        task_performances = hive.queen_genes[:, task_idx]
        
        # Calculate interaction matrix for this task (K-matrix style)
        for bee_i in 1:n_bees
            for bee_j in 1:n_bees
                if bee_i != bee_j
                    # Interaction rate: bee_j can suppress bee_i if bee_j performs better
                    performance_diff = task_performances[bee_j] - task_performances[bee_i]
                    if performance_diff > 0
                        # Interaction strength based on performance gap and lambda sensitivity
                        interaction_strength = config.interaction_rate * performance_diff * config.lambda_sensitivity
                        interact_propensities[bee_i, task_idx] += interaction_strength
                    end
                end
            end
        end
    end
    
    total_propensity = sum(train_propensities) + sum(interact_propensities)
    
    return train_propensities, interact_propensities, total_propensity
end

"""
    select_multitask_event(train_propensities::Matrix{Float64}, 
                          interact_propensities::Matrix{Float64}, 
                          total_propensity::Float64,
                          current_time::Float64)

Select which event occurs next using Gillespie algorithm mechanics.

# Arguments
- `train_propensities::Matrix{Float64}`: Training rates [bee, task]
- `interact_propensities::Matrix{Float64}`: Interaction rates [bee, task] 
- `total_propensity::Float64`: Sum of all propensities
- `current_time::Float64`: Current simulation time

# Returns
- `event::MultiTaskEvent`: Selected event with all details
- `time_step::Float64`: Time until this event occurs

Uses standard Gillespie selection: exponential time step, weighted random event choice.
"""
function select_multitask_event(train_propensities::Matrix{Float64}, 
                                interact_propensities::Matrix{Float64}, 
                                total_propensity::Float64,
                                current_time::Float64)
    
    # Step 1: Calculate time until next event (exponential distribution)
    time_step = rand(Exponential(1.0 / total_propensity))
    event_time = current_time + time_step
    
    # Step 2: Select which event occurs (weighted random selection)
    selection_value = rand() * total_propensity
    cumulative_propensity = 0.0
    
    n_bees, n_tasks = size(train_propensities)
    
    # Check training events first
    for task_idx in 1:n_tasks
        for bee_idx in 1:n_bees
            cumulative_propensity += train_propensities[bee_idx, task_idx]
            if selection_value <= cumulative_propensity
                return MultiTaskEvent(:train, bee_idx, task_idx, nothing, event_time), time_step
            end
        end
    end
    
    # Check interaction events  
    for task_idx in 1:n_tasks
        for bee_idx in 1:n_bees
            cumulative_propensity += interact_propensities[bee_idx, task_idx]
            if selection_value <= cumulative_propensity
                # For interactions, we need to find the dominant bee
                # This is simplified - in practice you might want more sophisticated selection
                partner_bee = find_dominant_partner(bee_idx, task_idx, train_propensities)
                return MultiTaskEvent(:interact, bee_idx, task_idx, partner_bee, event_time), time_step
            end
        end
    end
    
    # Fallback (should not reach here with proper propensity calculation)
    @warn "Event selection fallback triggered - this indicates a bug in propensity calculation"
    return MultiTaskEvent(:train, 1, 1, nothing, event_time), time_step
end

"""
    find_dominant_partner(subdominant_bee::Int, task_idx::Int, train_propensities::Matrix{Float64})

Find a dominant bee for interaction based on performance.
Simplified version - returns a random bee with higher performance.
"""
function find_dominant_partner(subdominant_bee::Int, task_idx::Int, queen_genes::Matrix{Float64})
    n_bees = size(queen_genes, 1)
    subdominant_performance = queen_genes[subdominant_bee, task_idx]
    
    # Find bees with higher performance on this task
    potential_dominants = Int[]
    for bee_idx in 1:n_bees
        if bee_idx != subdominant_bee && queen_genes[bee_idx, task_idx] > subdominant_performance
            push!(potential_dominants, bee_idx)
        end
    end
    
    # If no dominant bees found, return random bee (or could return nothing)
    if isempty(potential_dominants)
        candidates = [i for i in 1:n_bees if i != subdominant_bee]
        return isempty(candidates) ? subdominant_bee : rand(candidates)
    end
    
    return rand(potential_dominants)
end

"""
    execute_multitask_event!(event::MultiTaskEvent, hive::MultiTaskHive, 
                             train_loaders::Dict, test_loaders::Dict)

Execute the selected event and update the hive state.

# Arguments
- `event::MultiTaskEvent`: The event to execute
- `hive::MultiTaskHive`: Hive to modify
- `train_loaders::Dict`: Training data loaders by dataset name
- `test_loaders::Dict`: Test data loaders by dataset name

# Side Effects
- Updates hive state (queen_genes, train_counts, etc.)
- Updates current_time in hive

# Returns
- `event_result::Dict`: Information about what happened during execution
"""
function execute_multitask_event!(event::MultiTaskEvent, hive::MultiTaskHive, 
                                  train_loaders::Dict, test_loaders::Dict)
    
    # Update hive time
    hive.current_time = event.timestamp
    
    result = Dict{String, Any}()
    
    if event.event_type == :train
        # Execute training event
        dataset_name = hive.config.dataset_names[event.task_idx]
        train_loader = train_loaders[dataset_name]
        test_loader = test_loaders[dataset_name]
        
        # Use our training function
        training_loss, accuracy = train_bee_on_task!(hive, event.bee_idx, event.task_idx, 
                                                    train_loader, test_loader)
        
        result["type"] = "training"
        result["bee_idx"] = event.bee_idx
        result["task_idx"] = event.task_idx
        result["task_name"] = dataset_name
        result["training_loss"] = training_loss
        result["accuracy"] = accuracy
        result["train_count"] = hive.train_counts[event.bee_idx, event.task_idx]
        
    elseif event.event_type == :interact
        # Execute interaction event (punishment/suppression)
        subdominant_bee = event.bee_idx
        dominant_bee = event.partner_bee_idx
        task_idx = event.task_idx
        
        # Apply interaction effect (simplified: reset neural network)
        # In practice, this could be more sophisticated (weight perturbation, etc.)
        hive.brains[subdominant_bee] = hive.config.model_template()
        
        # Reset performance for this bee on this task
        hive.queen_genes[subdominant_bee, task_idx] = 0.0
        
        # Mark suppression state
        hive.suppressed_tasks[subdominant_bee, task_idx] = true
        hive.suppression_start_times[subdominant_bee, task_idx] = event.timestamp
        
        result["type"] = "interaction"
        result["subdominant_bee"] = subdominant_bee
        result["dominant_bee"] = dominant_bee
        result["task_idx"] = task_idx
        result["task_name"] = hive.config.dataset_names[task_idx]
        result["new_performance"] = hive.queen_genes[subdominant_bee, task_idx]
        
    else
        @warn "Unknown event type: $(event.event_type)"
        result["type"] = "unknown"
    end
    
    return result
end

"""
    multitask_gillespie_step!(hive::MultiTaskHive, train_loaders::Dict, test_loaders::Dict)

Execute one step of the multi-task Gillespie simulation.

# Arguments
- `hive::MultiTaskHive`: Current hive state (modified in place)
- `train_loaders::Dict`: Training data loaders
- `test_loaders::Dict`: Test data loaders

# Returns
- `event::MultiTaskEvent`: The event that was executed
- `result::Dict`: Details about the event execution

This is the core function that advances the simulation by one stochastic event.
"""
function multitask_gillespie_step!(hive::MultiTaskHive, train_loaders::Dict, test_loaders::Dict)
    
    # Calculate propensities for all possible events
    train_props, interact_props, total_prop = calculate_multitask_propensities(hive, hive.config)
    
    # Select which event occurs next
    event, time_step = select_multitask_event(train_props, interact_props, total_prop, hive.current_time)
    
    # Execute the selected event
    result = execute_multitask_event!(event, hive, train_loaders, test_loaders)
    
    return event, result
end