function save_states_csv(results::NamedTuple, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Prepare data - pure simulation data only
    n_time_points, n_bees, n_tasks = size(results.performance_history)
    data = []
    
    for t in 1:n_time_points
        for bee in 1:n_bees
            for task in 1:n_tasks
                task_value = results.performance_history[t, bee, task]
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
    n_time_points, n_bees, n_tasks = size(results.losses_history)
    data = []
    
    for t in 1:n_time_points
        for bee in 1:n_bees
            for task in 1:n_tasks
                loss_value = results.losses_history[t, bee, task]
                push!(data, (epoch=t-1, bee_id=bee, task_id=task, loss=loss_value))
            end
        end
    end
    
    # Convert to DataFrame and save with fixed filename
    df = DataFrame(data)
    
    filepath = joinpath(output_dir, "losses.csv")
    CSV.write(filepath, df)
    
    println("Loss evolution saved to: $filepath")
    return df
end

function save_events_csv(results::NamedTuple, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    # Prepare data - handle 4D arrays with structure [epochs, bee1, bee2, tasks]
    n_epochs, n_bee1, n_bee2, n_tasks = size(results.production_count)
    data = []
    
    for t in 1:n_epochs
        time_value = t
        for bee1 in 1:n_bee1
            for bee2 in 1:n_bee2
                for task in 1:n_tasks
                    production_value = results.production_count[t, bee1, bee2, task]
                    suppression_value = results.suppression_count[t, bee1, bee2, task]
                    push!(data, (epoch=t-1, time=time_value, bee1_id=bee1, bee2_id=bee2, task_id=task, 
                            production_count=production_value,
                            suppression_count=suppression_value))
                end
            end
        end
    end
    
    # Convert to DataFrame and save with fixed filename
    df = DataFrame(data)

    df = filter(row -> row.production_count > 0 || row.suppression_count > 0, df)
    
    filepath = joinpath(output_dir, "events.csv")
    CSV.write(filepath, df)
    
    println("Events saved to: $filepath")
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

function save_task_mapping(task_mapping::Dict{Int, String}, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    filepath = joinpath(output_dir, "task_index_mapping.json")
    open(filepath, "w") do io
        JSON3.pretty(io, task_mapping)
    end
    
    println("Task index mapping saved to: $filepath")
    return task_mapping
end

function save_log(log::GillespieEventLog, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    filepath = joinpath(output_dir, "event_log.json")
    open(filepath, "w") do io
        JSON3.pretty(io, log)
    end
    
    println("Event log saved to: $filepath")
    return log
end

function log_to_dataframe(log::GillespieEventLog)
    n = length(log.time)
    n_tasks = length(log.accuracies[1])
    accuracy_cols = [Symbol("task_$i") for i in 1:n_tasks]

    # Initialize columns for accuracies
    accuracy_data = [getindex.(log.accuracies, i) for i in 1:n_tasks]

    df = DataFrame(
        time = log.time,
        bee1_id = log.bee1_id,
        bee2_id = log.bee2_id,
        task_id = log.task_id,
    )

    for (colname, coldata) in zip(accuracy_cols, accuracy_data)
        df[!, colname] = coldata
    end

    return df
end

function save_log_df(df::DataFrame, output_dir::String)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    filepath = joinpath(output_dir, "event_log.csv")
    CSV.write(filepath, df)
    
    println("Event log DataFrame saved to: $filepath")
    return df
end