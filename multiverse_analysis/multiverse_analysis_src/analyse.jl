using Base.Threads
using Distributed
using DataFrames

function compute_analysis_df(states_df::DataFrame)
    #results = Vector{DataFrame}(undef, n_runs)
    dataframes = DataFrame[]
    epochs = unique(sort(states_df.epoch))
    for t in epochs
        epoch_states = states_df[states_df.epoch .== t, :]
        analysis_epoch_df = compute_analysis_epoch(epoch_states)
        push!(dataframes, analysis_epoch_df)
    end
    return vcat(dataframes...)
end

function compute_analysis_epoch(epoch_states)
    n_tasks = length(unique(epoch_states.task_id))
    epoch = epoch_states.epoch[1]
    df = DataFrame(
        epoch = Vector{Int}(undef, n_tasks),
        task_id = Vector{Int}(undef, n_tasks),
        score = Vector{Float64}(undef, n_tasks),
        best_bee_id = Vector{Int}(undef, n_tasks),
        best_task_value = Vector{Float64}(undef, n_tasks)
    )    

    for task_id in 1:n_tasks
        task_states = epoch_states[epoch_states.task_id .== task_id, :]
        max_task_value = maximum(task_states.task_value)
        df[task_id, :epoch] = epoch
        df[task_id, :task_id] = task_id
        df[task_id, :score] = axis_score(task_states.task_value)
        df[task_id, :best_bee_id] = task_states[task_states.task_value .== max_task_value, :bee_id][1]
        df[task_id, :best_task_value] = max_task_value
    end

    return df
end

function axis_score(n_values, eps=1e-9)
    n_max = maximum(n_values)
    n_sum = sum(n_values)
    if n_sum == 0
        return 0.0
    end
    n_second_highest = sort(n_values)[end-1]
    frac = n_max/n_sum
    dominance = (n_max - n_second_highest)/(n_max + eps)
    axis_score = frac * dominance
    return axis_score
end

function analyze_and_save_run(run_path::String)
    states = load_states(joinpath(run_path, "states.csv"))
    analysis_df = compute_analysis_df(states)
    CSV.write(joinpath(run_path, "analysis.csv"), analysis_df)
    return analysis_df
end

function analyze_and_save_runs_threaded(data_path::String)
    run_dirs = filter(isdir, readdir(data_path, join=true))
    Threads.@threads for run_dir in run_dirs
        analyze_and_save_run(run_dir)
    end
end

function find_best_run(vector_of_runs; run_type=:processed, test_type=:last_epoch)
    best_total = -Inf
    best_index = 0
    best_run = nothing
    total_task_value = 0.0
    
    for (i, run) in enumerate(vector_of_runs)
        # Get the analysis dataframe
        df = nothing
        if run_type == :processed
            df = run.analysis_df
        elseif run_type == :raw
            df = run.states
        else
            error("Unknown run_type: $run_type")
        end
        
        if test_type == :last_epoch
            # Find the last epoch
            last_epoch = maximum(df.epoch)
            
            # Filter to last epoch and sum all task_values
            last_epoch_data = filter(row -> row.epoch == last_epoch, df)
            if run_type == :processed
                total_task_value = sum(last_epoch_data.best_task_value)
            elseif run_type == :raw
                total_task_value = sum(last_epoch_data.task_value)
            else
                error("Unknown run_type: $run_type")
            end
        elseif test_type == :run_average
            # Calculate average task value for each task over all epochs, then sum
            if run_type == :processed
                # Group by task_id and calculate mean of best_task_value across all epochs
                task_averages = combine(groupby(df, :task_id), :best_task_value => mean => :avg_task_value)
                total_task_value = sum(task_averages.avg_task_value)
            elseif run_type == :raw
                # Group by task_id and calculate mean of task_value across all epochs
                task_averages = combine(groupby(df, :task_id), :task_value => mean => :avg_task_value)
                total_task_value = sum(task_averages.avg_task_value)
            else
                error("Unknown run_type: $run_type")
            end
        else
            error("Unknown test_type: $test_type")
        end
        
        # Check if this is the best so far
        if total_task_value > best_total
            best_total = total_task_value
            best_index = i
            best_run = run
        end
    end
    
    return (index=best_index, run=best_run, total_task_value=best_total)
end
