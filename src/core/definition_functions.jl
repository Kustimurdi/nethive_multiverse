# Hive Access Functions

"""
    get_bee_brain(hive::MultiTaskHive, bee_idx::Int)

Get the neural network for a specific bee.

# Returns
- `Flux.Chain`: Neural network for the bee
"""
function get_bee_brain(hive::MultiTaskHive, bee_idx::Int)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("Bee index must be between 1 and $(hive.n_bees)"))
    end
    return hive.brains[bee_idx]
end

"""
    get_bee_performance(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)

Get the performance (queen gene) of a specific bee on a specific task.

# Returns
- `Float64`: Performance value in [0,1]
"""
function get_bee_performance(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("Bee index must be between 1 and $(hive.n_bees)"))
    end
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("Task index must be between 1 and $(hive.n_tasks)"))
    end
    return hive.queen_genes[bee_idx, task_idx]
end

"""
    set_bee_performance!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, performance::Float64)

Set the performance (queen gene) of a specific bee on a specific task.

# Arguments
- `hive::MultiTaskHive`: The hive
- `bee_idx::Int`: Bee index (1-indexed)
- `task_idx::Int`: Task index (1-indexed)  
- `performance::Float64`: Performance value (will be clamped to [0,1])
"""
function set_bee_performance!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, performance::Float64)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("Bee index must be between 1 and $(hive.n_bees)"))
    end
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("Task index must be between 1 and $(hive.n_tasks)"))
    end
    hive.queen_genes[bee_idx, task_idx] = clamp(performance, 0.0, 1.0)
end

"""
    is_task_suppressed(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)

Check if a specific task is currently suppressed for a bee.

# Returns
- `Bool`: True if task is suppressed
"""
function is_task_suppressed(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("Bee index must be between 1 and $(hive.n_bees)"))
    end
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("Task index must be between 1 and $(hive.n_tasks)"))
    end
    return hive.suppressed_tasks[bee_idx, task_idx]
end

"""
    suppress_task!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, current_time::Float64)

Suppress a specific task for a bee.

# Arguments
- `hive::MultiTaskHive`: The hive
- `bee_idx::Int`: Bee index (1-indexed)
- `task_idx::Int`: Task index (1-indexed)
- `current_time::Float64`: Current simulation time
"""
function suppress_task!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, current_time::Float64)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("Bee index must be between 1 and $(hive.n_bees)"))
    end
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("Task index must be between 1 and $(hive.n_tasks)"))
    end
    hive.suppressed_tasks[bee_idx, task_idx] = true
    hive.suppression_start_times[bee_idx, task_idx] = current_time
end

"""
    resolve_suppression!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)

Resolve suppression for a specific task on a bee.

# Arguments
- `hive::MultiTaskHive`: The hive
- `bee_idx::Int`: Bee index (1-indexed)
- `task_idx::Int`: Task index (1-indexed)
"""
function resolve_suppression!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("Bee index must be between 1 and $(hive.n_bees)"))
    end
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("Task index must be between 1 and $(hive.n_tasks)"))
    end
    hive.suppressed_tasks[bee_idx, task_idx] = false
    hive.suppression_start_times[bee_idx, task_idx] = 0.0
end

"""
    increment_train_count!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)

Increment the training count for a bee on a specific task.

# Arguments
- `hive::MultiTaskHive`: The hive
- `bee_idx::Int`: Bee index (1-indexed)
- `task_idx::Int`: Task index (1-indexed)
"""
function increment_train_count!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("Bee index must be between 1 and $(hive.n_bees)"))
    end
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("Task index must be between 1 and $(hive.n_tasks)"))
    end
    hive.train_counts[bee_idx, task_idx] += 1
end

# Matrix Operations for Analysis

"""
    get_all_performances(hive::MultiTaskHive)

Get the full performance matrix for all bees and all tasks.

# Returns
- `Matrix{Float64}`: Performance matrix [n_bees × n_tasks] with values in [0,1]
"""
function get_all_performances(hive::MultiTaskHive)
    return copy(hive.queen_genes)
end

"""
    get_task_performances(hive::MultiTaskHive, task_idx::Int)

Get performance of all bees on a specific task.

# Returns
- `Vector{Float64}`: Performance values for all bees on the task
"""
function get_task_performances(hive::MultiTaskHive, task_idx::Int)
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("Task index must be between 1 and $(hive.n_tasks)"))
    end
    return hive.queen_genes[:, task_idx]
end

"""
    get_bee_performances(hive::MultiTaskHive, bee_idx::Int)

Get performance of a specific bee on all tasks.

# Returns
- `Vector{Float64}`: Performance values for the bee on all tasks
"""
function get_bee_performances(hive::MultiTaskHive, bee_idx::Int)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("Bee index must be between 1 and $(hive.n_bees)"))
    end
    return hive.queen_genes[bee_idx, :]
end