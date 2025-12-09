function plot_bee_all_tasks(df::DataFrame, bee_id;
                                  ax::Union{Nothing,Makie.Axis}=nothing,
                                  figsize=(1200,400),
                                  axpos::Tuple{Int,Int}=(1,1),
                                  linewidth=2,
                                  markersize=10,
                                  mapping::Union{Nothing,Dict{Any,String}}=nothing, 
                                  add_markers::Bool=true, 
                                  add_legend::Bool=true)

    circle_with_hole = BezierPath([
        MoveTo(Point(1, 0)),
        EllipticalArc(Point(0, 0), 1, 1, 0, 0, 2pi),
        MoveTo(Point(0.5, 0.5)),
        LineTo(Point(0.5, -0.5)),
        LineTo(Point(-0.5, -0.5)),
        LineTo(Point(-0.5, 0.5)),
        ClosePath(),
    ])


    tasks = sort(unique(df.task_id))
    filter!(e -> e != 0, tasks)
    
    created_fig = false
    if ax !== nothing
        ax = ax
    else
        created_fig = true
        fig = Makie.Figure(size = (figsize[1], figsize[2]))
        ax = Makie.Axis(fig[axpos[1], axpos[2]], 
                xlabel = "Time", 
                ylabel = "Accuracy",
                title = "Evolution of Accuracy for Bee $bee_id Across All Tasks")
    end
    
    # Safely build a list of colors: handle zero tasks and wrap the palette if more tasks than colors.
    palette = Makie.wong_colors()
    n_tasks = length(tasks)
    if n_tasks == 0
        colors = Any[]
    else
        colors = [palette[(i - 1) % length(palette) + 1] for i in 1:n_tasks]
    end
    
    # Plot individual points with markers
    training_df = filter(row -> row.bee1_id == bee_id && row.bee1_id == row.bee2_id, df)
    suppression_df = filter(row -> row.bee1_id == bee_id && row.bee1_id != row.bee2_id, df)
    suppression_df = filter(row -> row.bee2_id != 0, suppression_df)

    for (i, task) in enumerate(tasks)
        col_name = Symbol("task_$(task)")
        task_data = filter(row -> row.bee1_id == bee_id, df)
        ydata = Float64.(task_data[!, col_name])

        if mapping !== nothing && haskey(mapping, task)
            task_label = mapping[task]
        else
            task_label = "Task $task"
        end
        
        # Plot main line
        Makie.lines!(ax, task_data.time, ydata, 
               color = colors[i], linewidth = linewidth, label = task_label)
        
        # Plot training events
        train_rows = training_df[training_df.task_id .== task, :]
        if nrow(train_rows) > 0 && add_markers
            ytrain = Float64.(train_rows[!, col_name])
            Makie.scatter!(ax, train_rows.time, ytrain, alpha = 0.2,
                    color = :green, markersize = markersize+2, marker = :circle) #:green
                    #label = "Training Events")
        end
        # Plot suppression events
        suppr_rows = suppression_df[suppression_df.task_id .== task, :]
        if nrow(suppr_rows) > 0 && add_markers
            ysuppr = Float64.(suppr_rows[!, col_name])
            Makie.scatter!(ax, suppr_rows.time, ysuppr, alpha = 0.2,
                    color = :red, markersize = markersize+4, marker = :xcross) #:red
                    #label = "Suppression Events")
        end
    end

    # Add legend
    if add_legend
        axislegend(ax, position = :lt)
    end

    # Add grid
    ax.xgridvisible = true
    ax.ygridvisible = true

    # Return the Figure if we created one, otherwise return the Axis so callers
    # that provided an Axis can continue drawing into it.
    if created_fig
        return fig
    end

    return ax
end

function plot_task_all_bees(df::DataFrame, task_id;
                            ax::Union{Nothing,Makie.Axis}=nothing,
                            axpos::Tuple{Int,Int}=(1,1),
                            add_legend::Bool=true,
                            figsize=(1200,400),
                            linewidth=2,
                            markersize=10,
                            mapping::Union{Nothing,Dict{Any,String}}=nothing,
                            add_markers::Bool=true)
    col_name = Symbol("task_$(task_id)")
    created_fig = false
    if ax !== nothing
        ax = ax
    else
        created_fig = true
        fig = Makie.Figure(size = (figsize[1], figsize[2]))
        ax = Makie.Axis(fig[axpos[1], axpos[2]], 
                xlabel = "Time", 
                ylabel = "Accuracy",
                title = "Evolution of Accuracy for Bee $task_id Across All Tasks")
    end
    
    bees = sort(unique(df.bee1_id))
    colors = Makie.wong_colors()[1:length(bees)]

    training_df = filter(row -> row.bee1_id == row.bee2_id, df)
    suppression_df = filter(row -> row.bee1_id != row.bee2_id, df)
    
    for (i, bee) in enumerate(bees)
        bee_data = filter(row -> row.bee1_id == bee, df)
        ydata = Float64.(bee_data[!, col_name])
        
        if mapping !== nothing && haskey(mapping, task_id)
            task_label = mapping[task_id]
        else
            task_label = "Task $task_id"
        end
        label = "Bee $bee - $task_label"
        
        # Plot main line
        Makie.lines!(ax, bee_data.time, ydata, 
               color = colors[i], linewidth = linewidth, label = label)

        # Plot training events
        train_rows = training_df[(training_df.task_id .== task_id) .& (training_df.bee1_id .== bee), :]
        if nrow(train_rows) > 0 && add_markers
            ytrain = Float64.(train_rows[!, col_name])
            Makie.scatter!(ax, train_rows.time, ytrain, 
                    color = :black, markersize = markersize+2, marker = :rect, 
                    label = "Training Events")
        end
        # Plot suppression events
        suppr_rows = suppression_df[(suppression_df.task_id .== task_id) .& (suppression_df.bee1_id .== bee), :]
        if nrow(suppr_rows) > 0 && add_markers
            ysuppr = Float64.(suppr_rows[!, col_name])
            Makie.scatter!(ax, suppr_rows.time, ysuppr, 
                    color = Makie.RGBAf0(1.0, 0.0, 0.0, 0.4), markersize = markersize+2, 
                    marker = :utriangle, label = "Suppression Events")
        end
        
    end

    # Add legend
    if add_legend
        axislegend(ax, position = :lt)
    end
    
    # Add grid
    ax.xgridvisible = true
    ax.ygridvisible = true
    
    # Return the Figure if we created it, else return the Axis so callers can keep drawing
    if created_fig
        return fig
    end

    return ax
end

function plot_best_performance_of_task(df::DataFrame, task_id;
                                        figsize=(1200,800),
                                        color=:blue,
                                        ax::Union{Nothing,Makie.Axis}=nothing,
                                        axpos::Tuple{Int,Int}=(1,1),
                                        linewidth=2,
                                        markersize=10)
    col_name = Symbol("task_$(task_id)")
    created_fig = false
    if ax !== nothing
        ax = ax
    else
        created_fig = true
        fig = Makie.Figure(size = (figsize[1], figsize[2]))
        ax = Makie.Axis(fig[axpos[1], axpos[2]], 
                xlabel = "Epoch", 
                ylabel = "Best Task Value",
                title = "Evolution of Best Task Values with Bee Changes for Task $task_id")
    end

    # Use flexible containers: bee IDs in the log may be Float64 or Int64
    best_bee_id = Any[]
    best_task_value = Float64[]
    times = Float64[]
    current_best_value = -Inf
    current_best_bee = nothing
    
    for row in eachrow(df)
        time = row.time
        bee1 = row.bee1_id
        bee2 = row.bee2_id
        task_value = Float64(row[col_name])
        
        if task_value > current_best_value
            current_best_value = task_value
            current_best_bee = bee1
        end
        
        push!(times, float(time))
        push!(best_task_value, current_best_value)
        push!(best_bee_id, current_best_bee)
    end

    Makie.lines!(ax, times, best_task_value, 
               color = color, linewidth = linewidth, label = "Task $task_id")
    # Find bee change points
    bee_changes = []
    for j in 2:length(best_bee_id)
        if best_bee_id[j] != best_bee_id[j-1]
            push!(bee_changes, j)
        end
    end

    # Mark bee changes with scatter points
    if !isempty(bee_changes)
        change_times = times[bee_changes]
        change_values = best_task_value[bee_changes]
        Makie.scatter!(ax, change_times, change_values, 
                color = :red, marker = :circle, markersize = markersize,
                strokecolor = :white, strokewidth = 1)
    end

    # Optionally, add text annotations for bee IDs at change points
    for idx in bee_changes
        time = times[idx]
        value = best_task_value[idx]
        bee_id = best_bee_id[idx]
        Makie.text!(ax, time, value, text = "B$bee_id", 
              fontsize = 8, offset = (5, 5), color = :black)
    end

    # Add legend
    axislegend(ax, position = :lt) 

    # Add grid
    ax.xgridvisible = true
    ax.ygridvisible = true

    # Return the Figure if we created it, else return the Axis so callers can keep drawing
    if created_fig
        return fig
    end

    return ax
end


