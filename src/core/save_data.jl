function save_states_csv(results::NamedTuple, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Prepare data - pure simulation data only
    n_bees, n_tasks, n_time_points = size(results.performance_history)
    data = []
    
    for t in 1:n_time_points
        for bee in 1:n_bees
            for task in 1:n_tasks
                task_value = results.performance_history[bee, task, t]
                push!(data, (epoch=(t-1), bee_id=bee, task_id=task, accuracies=task_value))
            end
        end
    end
    
    # Convert to DataFrame and save with fixed filename
    df = DataFrame(data)
    
    filepath = joinpath(output_dir, "states.csv")
    CSV.write(filepath, df)
    
    println("Task evolution saved to: $filepath")
    return df
end

function save_losses_csv(results::NamedTuple, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Prepare data - pure simulation data only
    n_bees, n_tasks, n_time_points = size(results.losses_history)
    data = []
    
    for t in 1:n_time_points
        for bee in 1:n_bees
            for task in 1:n_tasks
                loss_value = results.losses_history[bee, task, t]
                push!(data, (epoch=t-1, bee_id=bee, task_id=task, loss=loss_value))
            end
        end
    end
    
    # Convert to DataFrame and save with fixed filename
    df = DataFrame(data)
    
    filepath = joinpath(output_dir, "losses.csv")
    CSV.write(filepath, df)
    
    println("Task evolution saved to: $filepath")
    return df
end

function save_events_csv(results::NamedTuple, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Prepare data - pure simulation data only
    n_bees, n_tasks, n_time_points = size(results.suppression_history)
    data = []
    
    for t in 1:n_time_points
        time_value = t
        for bee in 1:n_bees
            for task in 1:n_tasks
                production_value = results.production_count[bee, task, t]
                suppression_value = results.suppression_count[bee, task, t]
                push!(data, (time_step=t-1, time=time_value, bee_id=bee, task_id=task, 
                        production_count=production_value,
                        suppression_count=suppression_value))
            end
        end
    end
    
    # Convert to DataFrame and save with fixed filename
    df = DataFrame(data)
    
    filepath = joinpath(output_dir, "states.csv")
    CSV.write(filepath, df)
    
    println("Task evolution saved to: $filepath")
    return df
end

function save_simulation_results(results::NamedTuple, output_dir::String;
                                save_states::Bool=true,
                                save_events::Bool=false,
                                save_losses::Bool=false)
    
    println("Saving simulation results with clean separation...")
    saved_files = String[]
    
    # Save simulation data to CSV files (pure data, no metadata)
    if save_states
        save_states_csv(results, output_dir)
        push!(saved_files, "states.csv")
    end
    
    # Optional detailed data
    if save_events
        save_events_csv(results, output_dir)
        push!(saved_files, "events.csv")
    end
    
    if save_losses
        save_losses_csv(results, output_dir)
        push!(saved_files, "losses.csv")
    end
    
    return saved_files
end

function save_metadata_to_config(config::Dict, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Extract seed from config (it should already be there)
    seed = get(config, "seed", nothing)
    if seed === nothing
        error("Config dictionary must contain a 'seed' entry")
    end
    
    # Create complete config and metadata dictionary
    # Start with a copy of the original config
    metadata = copy(config)
    
    # Add/update run metadata
    metadata["random_seed"] = seed  # Ensure consistent naming
    
    filepath = joinpath(output_dir, "config.json")
    open(filepath, "w") do io
        JSON3.pretty(io, metadata)  # Use JSON3.pretty for consistency
    end
    
    println("Metadata and configuration saved to: $filepath")
    return metadata
end
