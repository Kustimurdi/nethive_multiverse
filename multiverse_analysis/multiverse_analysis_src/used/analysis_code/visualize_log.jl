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


using DataFrames

"""
Build a (agents × tasks) matrix of accuracies for a given snapshot time `t_snap`.
For each bee1_id, we pick the last event with time ≤ t_snap.
Returns: mat, xs (agent ids), ys (task nums)
"""
function acc_matrix_at_time(event_log::DataFrame, t_snap)
    # keep only events up to snapshot time
    df = event_log[event_log.time .<= t_snap, :]

    # last row per bee1_id (highest time within <= t_snap)
    log_last = unique(sort(df, [:bee1_id, :time], rev=[false, true]), :bee1_id)

    # keep only acc columns + bee1_id/time
    select!(log_last, Not(:bee2_id, :task_id))

    task_cols = names(log_last, r"^task_")

    log_long = stack(
        log_last,
        task_cols,
        variable_name = :task,
        value_name = :acc
    )
    log_long.task_num = parse.(Int, replace.(log_long.task, r"^task_" => ""))

    # matrix mapping
    xs = sort(unique(log_long.bee1_id))
    ys = sort(unique(log_long.task_num))
    xi = Dict(v => i for (i,v) in enumerate(xs))
    yi = Dict(v => i for (i,v) in enumerate(ys))

    mat = fill(NaN, length(xs), length(ys))
    for r in eachrow(log_long)
        mat[xi[r.bee1_id], yi[r.task_num]] = r.acc
    end

    return mat, xs, ys
end

function plot_acc_snapshots(event_log; t_snaps, nrows=2, ncols=3)

    @assert length(t_snaps) == nrows * ncols

    # -----------------------------
    # Typography (single source)
    # -----------------------------
    labelsize = 32
    ticklabelsize = 28
    titlesize = 26

    fig = Figure(resolution = (1600, 900))

    # -----------------------------
    # Precompute matrices
    # -----------------------------
    mats = Vector{Matrix{Float64}}(undef, length(t_snaps))
    xss  = Vector{Vector}(undef, length(t_snaps))
    yss  = Vector{Vector}(undef, length(t_snaps))

    for (k, t) in enumerate(t_snaps)
        mat, xs, ys = acc_matrix_at_time(event_log, t)
        mats[k] = mat
        xss[k] = xs
        yss[k] = ys
    end

    # Fixed color scale for accuracies
    clim = (0.0, 1.0)

    hms = Makie.Plot[]

    # -----------------------------
    # Plot grid
    # -----------------------------
    for k in 1:length(t_snaps)
        r = div(k-1, ncols) + 1
        c = mod(k-1, ncols) + 1

        ax = Axis(
            fig[r, c],
            xlabelvisible = false,
            ylabelvisible = false,
            xticklabelsize = ticklabelsize,
            yticklabelsize = ticklabelsize,
            title = "t = $(round(t_snaps[k]; digits=1))",
            titlesize = titlesize
        )

        xs = xss[k]
        ys = yss[k]

        hm = heatmap!(
            ax,
            1:length(xs),
            1:length(ys),
            mats[k];
            colormap = :viridis,
            colorrange = clim
        )

        ax.xticks = (1:length(xs), string.(xs))
        ax.yticks = (1:length(ys), string.(ys))

        push!(hms, hm)
    end

    # -----------------------------
    # Global axis labels
    # -----------------------------
    Label(
        fig[nrows+1, 1:ncols],
        #L"\text{agent id}",
        "agent id",
        fontsize = labelsize,
        tellwidth = false
    )

    Label(
        fig[1:nrows, 0],
        #L"\text{task id}",
        "task id",
        fontsize = labelsize,
        rotation = pi/2,
        tellheight = false
    )

    # -----------------------------
    # Shared colorbar
    # -----------------------------
    Colorbar(
        fig[1:nrows, ncols+1],
        hms[end],
        label = "accuracy",
        labelsize = labelsize,
        ticklabelsize = ticklabelsize
    )

    # Spacing tweaks
    colgap!(fig.layout, 16)
    rowgap!(fig.layout, 14)
    rowsize!(fig.layout, nrows+1, 40)

    return fig
end

function acc_matrix_final(event_log::DataFrame)
    t_final = maximum(event_log.time)
    return acc_matrix_at_time(event_log, t_final)  # returns mat, xs, ys
end

function acc_matrix_final_fixed(event_log::DataFrame, xs_all, ys_all)
    mat_local, xs, ys = acc_matrix_final(event_log)

    xi_all = Dict(v => i for (i,v) in enumerate(xs_all))
    yi_all = Dict(v => i for (i,v) in enumerate(ys_all))

    # remap local matrix onto global matrix
    mat = fill(NaN, length(xs_all), length(ys_all))
    xi = Dict(v => i for (i,v) in enumerate(xs))
    yi = Dict(v => i for (i,v) in enumerate(ys))

    for x in xs, y in ys
        mat[xi_all[x], yi_all[y]] = mat_local[xi[x], yi[y]]
    end
    return mat
end

function plot_final_across_reps_fixed(event_logs::Vector{DataFrame};
                                      ncols::Int=4,
                                      titles::Union{Nothing, Vector{String}}=nothing)

    n = length(event_logs)
    nrows = ceil(Int, n / ncols)

    # global axes sets
    xs_all = sort(unique(vcat([unique(df.bee1_id) for df in event_logs]...)))
    # tasks: infer from columns task_*
    # safer: take union across all logs
    tasknums = Int[]
    for df in event_logs
        cp = copy(df)
        select!(cp, Not(:task_id))
        cols = names(cp, r"^task_")
        append!(tasknums, parse.(Int, replace.(cols, r"^task_" => "")))
    end
    ys_all = sort(unique(tasknums))

    fig = Figure(resolution=(350*ncols + 200, 320*nrows + 120))
    clim = (0.0, 1.0)
    hms = Makie.Plot[]

    for k in 1:n
        r = div(k-1, ncols) + 1
        c = mod(k-1, ncols) + 1

        ax = Axis(fig[r, c],
            xlabelvisible=false, ylabelvisible=false,
            xticklabelsize=14, yticklabelsize=14, titlesize=16
        )

        mat = acc_matrix_final_fixed(event_logs[k], xs_all, ys_all)

        hm = heatmap!(ax, 1:length(xs_all), 1:length(ys_all), mat;
                      colormap=:viridis, colorrange=clim)

        ax.xticks = (1:length(xs_all), string.(xs_all))
        ax.yticks = (1:length(ys_all), string.(ys_all))

        ax.title = titles === nothing ? "rep $(k)" : titles[k]

        push!(hms, hm)
    end

    Label(fig[nrows+1, 1:ncols], "agent id", tellwidth=false)
    Label(fig[1:nrows, 0], "task id", rotation=pi/2, tellheight=false)

    Colorbar(fig[1:nrows, ncols+1], hms[end], label="accuracy")

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)

    return fig
end

function plot_cum_events_per_task(agg::DataFrame; ncols=3)
    tasks = sort(unique(agg.task_id))
    nt = length(tasks)
    nrows = ceil(Int, nt / ncols)

    fig = Figure(resolution=(450*ncols + 200, 320*nrows + 100))
    clim_dummy = (0,1)  # not used, just to keep layout similar to your other plots

    # global legend entries
    legend_handles = Dict{String, Any}()

    for (k, task) in enumerate(tasks)
        r = div(k-1, ncols) + 1
        c = mod(k-1, ncols) + 1

        ax = Axis(fig[r, c],
            title = "task $(task)",
            xlabelvisible = false,
            ylabelvisible = false,
            xticklabelsize = 18,
            yticklabelsize = 18,
            titlesize = 18,
        )

        sub = agg[agg.task_id .== task, :]

        for kind in ["train", "suppress"]
            s = sub[sub.kind .== kind, :]
            isempty(s) && continue

            # lines (optionally make suppress dashed)
            if kind == "train"
                h = lines!(ax, s.time, s.cum_n)
                legend_handles["training"] = h
            else
                h = lines!(ax, s.time, s.cum_n, linestyle = :dash)
                legend_handles["suppression"] = h
            end
        end
    end

    # global labels
    Label(fig[nrows+1, 1:ncols], "time", tellwidth=false, fontsize=24)
    Label(fig[1:nrows, 0], "cumulative # events", rotation=pi/2, tellheight=false, fontsize=24)

    # one shared legend (much better than per-subplot legends)
    Legend(fig[1:nrows, ncols+1],
        [legend_handles["training"], legend_handles["suppression"]],
        ["training", "suppression"]
    )

    colgap!(fig.layout, 14)
    rowgap!(fig.layout, 14)

    return fig
end
