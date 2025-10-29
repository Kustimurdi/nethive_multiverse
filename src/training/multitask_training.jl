"""
Multi-Task Training and Evaluation Module

This module provides training and evaluation functions adapted for the simplified MultiTaskHive structure
where all bees have identical neural networks and train on tasks with equal probability.
"""


"""
    perform_production!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, train_loader, test_loader)

Train a specific bee on a specific task for one training episode.

# Arguments
- `hive::MultiTaskHive`: The hive containing the bee
- `bee_idx::Int`: Index of the bee to train (1 to n_bees)
- `task_idx::Int`: Index of the task to train on (1 to n_tasks)
- `train_loader`: DataLoader for training data
- `test_loader`: DataLoader for testing (to compute performance)

# Returns
- `training_loss::Float64`: Average loss during training
- `accuracy::Float64`: Accuracy on test set after training (queen gene value)

# Side Effects
- Updates the bee's neural network weights
- Increments `hive.train_counts[bee_idx, task_idx]`
- Updates `hive.queen_genes[bee_idx, task_idx]` with new accuracy

This function represents one "training episode" in the Gillespie algorithm.
"""
function perform_production!(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, train_loader, test_loader)
    # Validate inputs
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("bee_idx must be between 1 and $(hive.n_bees)"))
    end
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("task_idx must be between 1 and $(hive.n_tasks)"))
    end
    
    # Get the bee's brain
    model = hive.brains[bee_idx]
    learning_rate = hive.config.learning_rate
    
    # Training phase
    training_loss = train_model!(model, train_loader; learning_rate=learning_rate)
    
    # Evaluation phase
    accuracy = calc_classification_accuracy(model, test_loader)
    
    # Update hive state
    hive.queen_genes[bee_idx, task_idx] = accuracy
    
    return training_loss, accuracy
end

function train_model!(model, train_loader; learning_rate)

    opt_state = Flux.setup(Flux.Adam(learning_rate), model)
    total_batch_loss = 0.0
    n_batches = 0
    
    for (x_batch, y_batch) in train_loader
        loss, grads = Flux.withgradient(model) do m
            Flux.Losses.logitcrossentropy(m(x_batch), y_batch)
        end
        Flux.update!(opt_state, model, grads[1])
        total_batch_loss += loss
        n_batches += 1
    end
    
    training_loss = total_batch_loss / max(n_batches, 1)
    return training_loss
end

"""
    calc_classification_accuracy(model, dataloader; num_batches::Int=typemax(Int))

Calculate classification accuracy on a dataset.

# Arguments
- `model`: Neural network model
- `dataloader`: DataLoader with (x, y) batches where y is one-hot encoded
- `num_batches`: Maximum number of batches to evaluate (default: all)

# Returns
- `accuracy::Float64`: Accuracy as a fraction in [0, 1]
"""
function calc_classification_accuracy(model, dataloader; num_batches::Int=typemax(Int))
    correct = 0
    total = 0
    
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        # Get predicted class indices (highest output)
        preds = Flux.onecold(model(x_batch))
        
        # Get true class indices from one-hot encoding
        truths = Flux.onecold(y_batch)
        
        correct += sum(preds .== truths)
        total += length(truths)
    end
    
    return correct / total
end

"""
    calc_classification_loss(model, dataloader; num_batches::Int=typemax(Int))

Calculate classification loss on a dataset.

# Arguments
- `model`: Neural network model  
- `dataloader`: DataLoader with (x, y) batches where y is one-hot encoded
- `num_batches`: Maximum number of batches to evaluate (default: all)

# Returns
- `loss::Float64`: Average logitcrossentropy loss
"""
function calc_classification_loss(model, dataloader; num_batches::Int=typemax(Int))
    total_loss = 0.0
    n_batches = 0
    
    loss_fn(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
    
    for (x_batch, y_batch) in Iterators.take(dataloader, num_batches)
        total_loss += loss_fn(x_batch, y_batch)
        n_batches += 1
    end
    
    return total_loss / max(n_batches, 1)
end

"""
    evaluate_bee_on_task(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, test_loader)

Evaluate a specific bee on a specific task without training.

# Arguments
- `hive::MultiTaskHive`: The hive containing the bee
- `bee_idx::Int`: Index of the bee to evaluate
- `task_idx::Int`: Index of the task to evaluate on
- `test_loader`: DataLoader for test data

# Returns
- `accuracy::Float64`: Accuracy on test set
- `loss::Float64`: Loss on test set

This function does not modify the hive state.
"""
function evaluate_bee_on_task(hive::MultiTaskHive, bee_idx::Int, task_idx::Int, test_loader)
    # Validate inputs
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("bee_idx must be between 1 and $(hive.n_bees)"))
    end
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("task_idx must be between 1 and $(hive.n_tasks)"))
    end
    
    model = hive.brains[bee_idx]
    accuracy = calc_classification_accuracy(model, test_loader)
    loss = calc_classification_loss(model, test_loader)
    
    return accuracy, loss
end


"""
    evaluate_all_bees_on_task(hive::MultiTaskHive, task_idx::Int, test_loader)

Evaluate all bees on a specific task.

# Arguments
- `hive::MultiTaskHive`: The hive
- `task_idx::Int`: Index of the task to evaluate on
- `test_loader`: DataLoader for test data

# Returns
- `accuracies::Vector{Float64}`: Accuracies for each bee (length n_bees)
- `losses::Vector{Float64}`: Losses for each bee (length n_bees)

This function does not modify the hive state.
"""
function evaluate_all_bees_on_task(hive::MultiTaskHive, task_idx::Int, test_loader)
    if task_idx < 1 || task_idx > hive.n_tasks
        throw(ArgumentError("task_idx must be between 1 and $(hive.n_tasks)"))
    end
    
    accuracies = Vector{Float64}(undef, hive.n_bees)
    losses = Vector{Float64}(undef, hive.n_bees)
    
    for bee_idx in 1:hive.n_bees
        accuracies[bee_idx], losses[bee_idx] = evaluate_bee_on_task(hive, bee_idx, task_idx, test_loader)
    end
    
    return accuracies, losses
end

"""
    evaluate_bee_on_all_tasks(hive::MultiTaskHive, bee_idx::Int, test_loaders::Dict)

Evaluate a specific bee on all tasks.

# Arguments
- `hive::MultiTaskHive`: The hive
- `bee_idx::Int`: Index of the bee to evaluate
- `test_loaders::Dict`: Dictionary mapping task names to test DataLoaders

# Returns
- `accuracies::Vector{Float64}`: Accuracies for each task (length n_tasks)
- `losses::Vector{Float64}`: Losses for each task (length n_tasks)

This function does not modify the hive state.
"""
function evaluate_bee_on_all_tasks(hive::MultiTaskHive, bee_idx::Int, loaders::Dict)
    if bee_idx < 1 || bee_idx > hive.n_bees
        throw(ArgumentError("bee_idx must be between 1 and $(hive.n_bees)"))
    end
    
    accuracies = Vector{Float64}(undef, hive.n_tasks)
    losses = Vector{Float64}(undef, hive.n_tasks)
    
    for task_idx in 1:hive.n_tasks
        dataset_name = hive.config.index_to_task_mapping[task_idx]
        test_loader = loaders[dataset_name]["test"]
        accuracies[task_idx], losses[task_idx] = evaluate_bee_on_task(hive, bee_idx, task_idx, test_loader)
    end
    
    return accuracies, losses
end

"""
    get_hive_performance_summary(hive::MultiTaskHive)

Get a summary of the current hive performance.

# Arguments
- `hive::MultiTaskHive`: The hive

# Returns
- `summary::Dict`: Dictionary with performance statistics

Contains:
- `mean_performance_per_task`: Average accuracy across all bees for each task
- `mean_performance_per_bee`: Average accuracy across all tasks for each bee
- `overall_mean_performance`: Overall average accuracy
- `total_training_episodes`: Total number of training episodes completed
- `training_episodes_per_task`: Total training episodes for each task
"""
function get_hive_performance_summary(hive::MultiTaskHive)
    mean_performance_per_task = vec(mean(hive.queen_genes, dims=1))
    mean_performance_per_bee = vec(mean(hive.queen_genes, dims=2))
    overall_mean_performance = mean(hive.queen_genes)
    total_training_episodes = sum(hive.train_counts)
    training_episodes_per_task = vec(sum(hive.train_counts, dims=1))
    
    return Dict(
        "mean_performance_per_task" => mean_performance_per_task,
        "mean_performance_per_bee" => mean_performance_per_bee,
        "overall_mean_performance" => overall_mean_performance,
        "total_training_episodes" => total_training_episodes,
        "training_episodes_per_task" => training_episodes_per_task,
        "current_epoch" => hive.current_epoch,
        "current_time" => hive.current_time
    )
end

"""
    update_all_queen_genes!(hive::MultiTaskHive, test_loaders::Dict)

Update all queen genes (performance matrix) by evaluating all bees on all tasks.

# Arguments
- `hive::MultiTaskHive`: The hive
- `test_loaders::Dict`: Dictionary mapping task names to test DataLoaders

# Side Effects
- Updates `hive.queen_genes` matrix with current accuracies

This is useful for periodic full evaluation of the hive state.
"""
function update_all_bees!(hive::MultiTaskHive, loaders::Dict)
    for bee_idx in 1:hive.n_bees
        for task_idx in 1:hive.n_tasks
            dataset_name = hive.config.index_to_task_mapping[task_idx]
            test_loader = loaders[dataset_name]["test"]
            accuracy, loss = evaluate_bee_on_task(hive, bee_idx, task_idx, test_loader)
            hive.queen_genes[bee_idx, task_idx] = accuracy
            hive.losses[bee_idx, task_idx] = loss
        end
    end
end
