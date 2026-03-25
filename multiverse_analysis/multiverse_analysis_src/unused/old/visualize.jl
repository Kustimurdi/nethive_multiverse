using CairoMakie
using DataFrames

function prepare_one_run(processed_data::NamedTuple)
    cfg = deepcopy(processed_data.config)
    scores = processed_data.scores
    leading_bees = processed_data.leading_bees
    run_path = processed_data.run_path
    final_time = maximum(scores.time)[1]
    final_scores = scores[scores.time .== final_time, :]
    col_names = names(select(final_scores, Not(:time)))
    score_vector = [final_scores[1, name] for name in col_names]
    run_score = prod(score_vector)
    cfg[!, :run_path] .= run_path
    cfg[!, :final_time] .= final_time
    cfg[!, :score] .= run_score
    final_leading_bees_row = leading_bees[leading_bees.time .== final_time, :]
    lb_df_names = names(select(final_leading_bees_row, Not(:time)))
    final_leading_bees_vector = [final_leading_bees_row[1, name] for name in lb_df_names]
    max_overlap = cfg.n_tasks[1] - length(unique(final_leading_bees_vector))
    cfg[!, :max_overlap] .= max_overlap
    return (config=cfg, scores=scores, leading_bees=leading_bees)
end

function prepare_heatmap_data(runs_data::Vector)
    df_list = DataFrame[]
    Threads.@threads for processed_data in runs_data
        prepared = prepare_one_run(processed_data)
        push!(df_list, prepared.config)
    end
    return vcat(df_list...)
end




function plot_best_task_evolution(df::DataFrame)
    # Get unique tasks
    tasks = sort(unique(df.task_id))
    
    # Create figure
    fig = Figure(resolution = (1000, 600))
    ax = Axis(fig[1, 1], 
              xlabel = "Epoch", 
              ylabel = "Best Task Value",
              title = "Evolution of Best Task Values Over Epochs")
    
    # Color palette for different tasks
    colors = Makie.wong_colors()[1:length(tasks)]
    
    for (i, task) in enumerate(tasks)
        # Filter data for this task
        task_data = filter(row -> row.task_id == task, df)
        task_data = sort(task_data, :epoch)
        
        # Plot the main line
        lines!(ax, task_data.epoch, task_data.best_task_value, 
               color = colors[i], linewidth = 2, label = "Task $task")
        
        # Find bee change points
        bee_changes = []
        for j in 2:nrow(task_data)
            if task_data.best_bee_id[j] != task_data.best_bee_id[j-1]
                push!(bee_changes, j)
            end
        end
        
        # Mark bee changes with scatter points
        if !isempty(bee_changes)
            change_epochs = task_data.epoch[bee_changes]
            change_values = task_data.best_task_value[bee_changes]
            scatter!(ax, change_epochs, change_values, 
                    color = colors[i], marker = :circle, markersize = 8,
                    strokecolor = :white, strokewidth = 1)
        end
        
        # Optionally, add text annotations for bee IDs at change points
        for idx in bee_changes
            epoch = task_data.epoch[idx]
            value = task_data.best_task_value[idx]
            bee_id = task_data.best_bee_id[idx]
            text!(ax, epoch, value, text = "B$bee_id", 
                  fontsize = 8, offset = (5, 5), color = colors[i])
        end
    end
    
    # Add legend
    axislegend(ax, position = :lt)
    
    # Add grid
    ax.xgridvisible = true
    ax.ygridvisible = true
    
    return fig
end

function plot_best_task_evolution_advanced(df::DataFrame)
    tasks = sort(unique(df.task_id))
    
    fig = Figure(resolution = (1200, 800))
    ax = Axis(fig[1, 1], 
              xlabel = "Epoch", 
              ylabel = "Best Task Value",
              title = "Evolution of Best Task Values with Bee Changes")
    
    colors = Makie.wong_colors()[1:length(tasks)]
    
    for (i, task) in enumerate(tasks)
        task_data = filter(row -> row.task_id == task, df)
        task_data = sort(task_data, :epoch)
        
        # Plot main line
        lines!(ax, task_data.epoch, task_data.best_task_value, 
               color = colors[i], linewidth = 3, label = "Task $task")
        
        # Create segments with different bee IDs
        current_bee = task_data.best_bee_id[1]
        segment_start = 1
        
        for j in 2:nrow(task_data)
            if task_data.best_bee_id[j] != current_bee
                # End current segment and start new one
                segment_end = j - 1
                
                # Add a vertical line to mark the change
                vlines!(ax, task_data.epoch[j], 
                       color = colors[i], alpha = 0.3, linestyle = :dash)
                
                # Add annotation with bee IDs
                mid_epoch = (task_data.epoch[segment_start] + task_data.epoch[segment_end]) / 2
                mid_value = (task_data.best_task_value[segment_start] + task_data.best_task_value[segment_end]) / 2
                
                text!(ax, mid_epoch, mid_value, 
                      text = "Bee $current_bee", 
                      fontsize = 10, 
                      color = colors[i],
                      align = (:center, :center),
                      backgroundcolor = (:white, 0.8))
                
                current_bee = task_data.best_bee_id[j]
                segment_start = j
            end
        end
        
        # Handle the last segment
        mid_epoch = (task_data.epoch[segment_start] + task_data.epoch[end]) / 2
        mid_value = (task_data.best_task_value[segment_start] + task_data.best_task_value[end]) / 2
        text!(ax, mid_epoch, mid_value, 
              text = "Bee $current_bee", 
              fontsize = 10, 
              color = colors[i],
              align = (:center, :center),
              backgroundcolor = (:white, 0.8))
    end
    
    axislegend(ax, position = :lt)
    ax.xgridvisible = true
    ax.ygridvisible = true
    
    return fig
end

# Function to plot all tasks for a single bee
function plot_bee_tasks(states_df, bee_id; plot_type=:line)
    # Filter data for the specific bee
    bee_data = filter(row -> row.bee_id == bee_id, states_df)
    
    acc_fig = Figure(resolution=(1000, 600))
    ax = Axis(acc_fig[1,1], 
              xlabel="Epoch",
              ylabel="Task Value (Accuracy)",
              title="Task Evolution for Bee $bee_id")
    
    # Get unique tasks for this bee and group by task
    tasks = sort(unique(bee_data.task_id))
    colors = Makie.wong_colors()[1:length(tasks)]
    
    # Plot each task as a separate line
    for (i, task) in enumerate(tasks)
        task_data = filter(row -> row.task_id == task, bee_data)
        sort!(task_data, :epoch)  # Ensure chronological order
        
        if plot_type == :scatter
            scatter!(ax, task_data.epoch, task_data.task_value, 
                     color = colors[i], markersize = 6, label = "Task $task")
        elseif plot_type == :line
            lines!(ax, task_data.epoch, task_data.task_value, 
                color = colors[i], linewidth = 2, label = "Task $task")
        else
            error("Unknown plot_type: $plot_type")
        end
    end
    
    # Add legend and grid
    axislegend(ax, position = :lt)
    ax.xgridvisible = true
    ax.ygridvisible = true
    
    return acc_fig
end