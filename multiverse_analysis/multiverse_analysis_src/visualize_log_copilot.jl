using CairoMakie, ColorSchemes, DataFrames, Statistics

"""
plot_specialization_heatmap(log::DataFrame; kwargs...)

Create a heatmap of final per-bee × per-task values and annotate each bee's best task.

Inputs
- log::DataFrame : event log with time, bee1_id, bee2_id, task_id and columns `task_N` (post-event values)
Keyword args
- task_prefix::String = "task_"         : prefix of task columns
- timecol::Symbol = :time               : time column
- bee1col::Symbol = :bee1_id
- bee2col::Symbol = :bee2_id
- show_values::Bool = false             : overlay numeric values in cells
- dominance_threshold::Real = 0.0       : minimum normalized dominance to consider a 'strong' specialization
- cmap = :viridis                       : colormap symbol or ColorScheme
- nan_color = RGBAf0(0.8,0.8,0.8,1.0)   : color to use for NaNs (no-data)
- fig_size = (800, 400)

Returns a NamedTuple:
(figure, axis, bees, task_ids, matrix, per_bee_best_task, per_task_top_bee, per_bee_spec_score)
"""
function plot_specialization_heatmap(log::DataFrame; task_prefix::String="task_",
                                     timecol::Symbol=:time, bee1col::Symbol=:bee1_id,
                                     bee2col::Symbol=:bee2_id,
                                     show_values::Bool=false,
                                     dominance_threshold::Real=0.0,
                                     cmap=:viridis,
                                     nan_color = RGBAf0(0.8,0.8,0.8,1.0),
                                     fig_size=(800,400))

    # 1) discover task ids
    task_cols = filter(c -> startswith(string(c), task_prefix), names(log))
    task_ids = Int[]
    for c in task_cols
        m = match(Regex("^" * task_prefix * raw"(\d+)$"), string(c))
        if m !== nothing
            push!(task_ids, parse(Int, m.captures[1]))
        end
    end
    sort!(task_ids)
    n_tasks = length(task_ids)

    if n_tasks == 0
        error("plot_specialization_heatmap: no task columns found with prefix=$(task_prefix)")
    end

    # 2) collect bees and build last-known states (scan in time order)
    bees = unique(vcat(log[!, bee1col], log[!, bee2col]))
    bees = collect(bees)  # preserve order (you may want to sort)
    n_bees = length(bees)

    last_known = Dict{Any, Dict{Int,Float64}}()
    # initialize
    for b in bees
        last_known[b] = Dict{Int,Float64}()  # we'll fill tasks, missing -> NaN later
    end

    sorted_log = sort(log, timecol)
    for row in eachrow(sorted_log)
        b = row[bee1col]
        for tid in task_ids
            col = Symbol(task_prefix * string(tid))
            if (hasproperty(row, col) || haskey(row, col)) && !ismissing(row[col])
                last_known[b][tid] = float(row[col])
            end
        end
    end

    # fill missing with NaN so matrix is rectangular
    for b in bees
        for tid in task_ids
            if !haskey(last_known[b], tid)
                last_known[b][tid] = NaN
            end
        end
    end

    # 3) build matrix rows=bees, cols=tasks (Matrix{Float64})
    M = Array{Float64}(undef, n_bees, n_tasks)
    for (i,b) in enumerate(bees)
        for (j,tid) in enumerate(task_ids)
            M[i,j] = last_known[b][tid]
        end
    end

    # 4) compute per-bee best task and per-bee spec score (normalized advantage)
    eps = 1e-9
    per_bee_best_task = Dict{Any,Int}()
    per_bee_best_value = Dict{Any,Float64}()
    per_bee_spec_score = Dict{Any,Float64}()
    for (i,b) in enumerate(bees)
        vals = M[i, :]
        valid_idx = findall(!isnan, vals)
        if isempty(valid_idx)
            per_bee_best_task[b] = -1
            per_bee_best_value[b] = NaN
            per_bee_spec_score[b] = NaN
            continue
        end
        # find best and compute advantage over mean of others
        best_rel = argmax(vals[valid_idx])
        best_idx = valid_idx[best_rel]
        best_val = vals[best_idx]
        others = [vals[k] for k in 1:length(vals) if k != best_idx && !isnan(vals[k])]
        per_bee_best_task[b] = task_ids[best_idx]
        per_bee_best_value[b] = best_val
        if isempty(others) || isnan(best_val)
            per_bee_spec_score[b] = NaN
        else
            per_bee_spec_score[b] = (best_val - mean(others)) / (best_val + eps)
        end
    end

    # 5) per-task top bee (for overlay) and dominance
    per_task_top_bee = Dict{Int,Any}()
    per_task_top_value = Dict{Int,Float64}()
    per_task_second_value = Dict{Int,Float64}()
    for (j,tid) in enumerate(task_ids)
        col_vals = [(bees[i], M[i,j]) for i in 1:n_bees if !isnan(M[i,j])]
        if isempty(col_vals)
            per_task_top_bee[tid] = nothing
            per_task_top_value[tid] = NaN
            per_task_second_value[tid] = NaN
            continue
        end
        sorted_col = sort(col_vals, by=x->x[2], rev=true)
        per_task_top_bee[tid] = sorted_col[1][1]
        per_task_top_value[tid] = sorted_col[1][2]
        per_task_second_value[tid] = length(sorted_col) >= 2 ? sorted_col[2][2] : NaN
    end

    # 6) plotting
    fig = Figure(resolution = fig_size)  # (Makie accepts `resolution=` on recent versions; if backend warns, replace with size=)
    ax = Axis(fig[1,1]; xlabel="task id", ylabel="bee", xticks=(1:n_tasks, string.(task_ids)),
              yticks=(1:n_bees, string.(bees)), width=fig_size[1], height=fig_size[2])

    # handle NaNs: map NaNs to a color by creating a colormap with nan_color
    # simplest: replace NaN with minimum-δ so they appear distinct; keep a mask for annotations
    finite_vals = filter(!isnan, vec(M))
    if isempty(finite_vals)
        # nothing to show
        heatmap!(ax, fill(0.0, size(M)); colormap=cmap)
    else
        vmin = minimum(finite_vals)
        vmax = maximum(finite_vals)
        # create display matrix where NaN -> vmin - delta
        delta = max((vmax - vmin) * 0.05, 1e-6)
        M_disp = copy(M)
        nanmask = isnan.(M_disp)
        M_disp[nanmask] .= vmin - delta
        # choose a colormap and plot
        hm = heatmap!(ax, 1:n_tasks, 1:n_bees, M_disp';
                      colormap = cmap, colorrange=(vmin - delta, vmax))
        # set NaN color by adding a colorbar and drawing a small rectangle legend manually is more work;
        # we annotate NaNs below if show_values
    end

    # optionally overlay numeric values
    if show_values
        for i in 1:n_bees, j in 1:n_tasks
            val = M[i,j]
            if !isnan(val)
                text!(ax, j, i, string(round(val, sigdigits=3)); align = (:center, :center), color=:white, fontsize=10)
            else
                text!(ax, j, i, "—"; align = (:center, :center), color=:black, fontsize=8)
            end
        end
    end

    # overlay markers for per-task top bee and per-bee best task:
    for (j, tid) in enumerate(task_ids)
        b = per_task_top_bee[tid]
        if b === nothing
            continue
        end
        bi = findfirst(isequal(b), bees)
        if bi === nothing
            continue
        end
        # draw circle marker on the top bee for this task
        scatter!(ax, [j], [bi]; markersize=10, color=:white, strokewidth=2, strokecolor=:black)
    end

    # highlight bees' best tasks (different marker if dominance small)
    for (i,b) in enumerate(bees)
        tid = per_bee_best_task[b]
        if tid == -1
            continue
        end
        j = findfirst(==(tid), task_ids)
        if j === nothing
            continue
        end
        spec = per_bee_spec_score[b]
        color = isnan(spec) ? RGBAf0(0.5,0.5,0.5,1.0) : (spec >= dominance_threshold ? :red : :orange)
        scatter!(ax, [j], [i]; markersize = 14, color = color, marker = :star5, strokewidth=1.5, strokecolor=:black)
    end

    Colorbar(fig[1,2], ax.scene.plots[1]; label="task value")  # attach colorbar

    fig.layoutgap = (10,10)

    display(fig)

    return (figure = fig,
            axis = ax,
            bees = bees,
            task_ids = task_ids,
            matrix = M,
            per_bee_best_task = per_bee_best_task,
            per_bee_spec_score = per_bee_spec_score,
            per_task_top_bee = per_task_top_bee,
            per_task_top_value = per_task_top_value)
end



using DataFrames
using Plots
using Statistics

# Helpers for plotting logs produced by the simulator.
# Functions:
# - task_columns(df; prefix="task_") -> Vector{Symbol}
# - plot_tasks_grid_for_bee(df, bee; kwargs...) -> Plots.Plot
# - plot_task_grid_across_bees(df, task; kwargs...) -> Plots.Plot
#
# Behaviour:
# - Automatically discovers task columns named like `task_1`, `task_2`, ...
# - Sorts rows by `time` before plotting
# - Highlights training events (bee1 == bee2) vs suppression events (bee1 != bee2)

function task_columns(df::DataFrame; prefix::String = "task_") # match columns like "task_1", "task_2" (prefix followed by digits)
    # construct pattern without letting `$` trigger Julia interpolation
    pat = Regex("^" * prefix * raw"\d+$")
    return filter(c -> !isnothing(match(pat, String(c))), names(df))
end

function _movavg(x::AbstractVector{T}, k::Int=5) where T<:Real
    if k <= 1 || length(x) <= k
        return Float64.(x)
    end
    n = length(x)
    pad = fld(k, 2)
    y = Array{Float64}(undef, n)
    for i in 1:n
        lo = max(1, i - pad); hi = min(n, i + pad)
        y[i] = mean(skipmissing(x[lo:hi]))
    end
    return y
end

# Plot a grid of subplots where each subplot is one task's time series for the given bee.
function plot_tasks_grid_for_bee(df::DataFrame, bee;
                                 beecol::Symbol = :bee1_id,
                                 otherbeecol::Symbol = :bee2_id,
                                 timecol::Symbol = :time,
                                 task_prefix::String = "task_",
                                 taskidcol::Symbol = :task_id,
                                 ncols::Int = 2,
                                 smooth::Bool = false,
                                 smooth_k::Int = 5,
                                 figsize = (900, 300))

    cols = task_columns(df; prefix=task_prefix)
    n = length(cols)
    if n == 0
        error("No task columns found with prefix='$(task_prefix)'")
    end

    rows = df[df[!, beecol] .== bee, :]
    if nrow(rows) == 0
        error("No rows found for $beecol == $bee")
    end
    sort!(rows, timecol)

    nrows = ceil(Int, n / ncols)
    plt = Plots.plot(layout = (nrows, ncols), size = (figsize[1], figsize[2]*nrows))

    for (i, col) in enumerate(cols)
        ax = i
        t = rows[!, timecol]
        y = Float64.(rows[!, col])
        if smooth
            yplot = _movavg(y, smooth_k)
        else
            yplot = y
        end

        Plots.plot!(plt[ax], t, yplot, label = String(col), lw=2)
        Plots.xlabel!(plt[ax], String(timecol)); Plots.ylabel!(plt[ax], String(col))

        # Mark events: training (bee1==bee2) and suppression (bee1!=bee2)
        # We mark only rows where the event's task equals this task index
        # Derive task index from column name like "task_3" => 3
        m = match(r"$(task_prefix)(\d+)", String(col))
        taskidx = m !== nothing ? parse(Int, m.captures[1]) : nothing
        if taskidx !== nothing && (taskidcol in names(rows))
            is_task = rows[!, taskidcol] .== taskidx
            is_train = is_task .& (rows[!, beecol] .== rows[!, otherbeecol])
            is_suppr = is_task .& (rows[!, beecol] .!= rows[!, otherbeecol])
            if any(is_train)
                scatter!(plt[ax], t[is_train], y[is_train], label = "train", m=:circle, ms=4, color=:green)
            end
            if any(is_suppr)
                scatter!(plt[ax], t[is_suppr], y[is_suppr], label = "suppress", m=:x, ms=4, color=:red)
            end
        end
    end

    return plt
end

# Plot a grid of subplots where each subplot is the same task for one bee.
function plot_task_grid_across_bees(df::DataFrame, task;
                                   beecol::Symbol = :bee1_id,
                                   otherbeecol::Symbol = :bee2_id,
                                   timecol::Symbol = :time,
                                   task_prefix::String = "task_",
                                   taskidcol::Symbol = :task_id,
                                   ncols::Int = 2,
                                   smooth::Bool = false,
                                   smooth_k::Int = 5,
                                   figsize = (900, 300))

    taskcol = task isa Int ? Symbol("$(task_prefix)$(task)") : Symbol(task)
    if !(taskcol in names(df))
        error("Task column $(taskcol) not found in DataFrame")
    end

    bees = sort(unique(df[!, beecol]))
    n = length(bees)
    if n == 0
        error("No bees found in column $(beecol)")
    end

    nrows = ceil(Int, n / ncols)
    plt = plot(layout = (nrows, ncols), size = (figsize[1], figsize[2]*nrows))

    for (i, b) in enumerate(bees)
        ax = i
        rows = df[df[!, beecol] .== b, :]
        if nrow(rows) == 0
            continue
        end
        sort!(rows, timecol)
        t = rows[!, timecol]
        y = Float64.(rows[!, taskcol])
        yplot = smooth ? _movavg(y, smooth_k) : y

        plot!(plt[ax], t, yplot, label = "bee $(b)", lw=2)
        xlabel!(plt[ax], String(timecol)); ylabel!(plt[ax], String(taskcol))

        # mark training/suppression events for this bee on this task
        if taskidcol in names(rows)
            is_task = rows[!, taskidcol] .== (match(r"$(task_prefix)(\d+)", String(taskcol)) !== nothing ? parse(Int, match(r"$(task_prefix)(\d+)", String(taskcol)).captures[1]) : -1)
            is_train = is_task .& (rows[!, beecol] .== rows[!, otherbeecol])
            is_suppr = is_task .& (rows[!, beecol] .!= rows[!, otherbeecol])
            if any(is_train)
                scatter!(plt[ax], t[is_train], y[is_train], label = "train", m=:circle, ms=4, color=:green)
            end
            if any(is_suppr)
                scatter!(plt[ax], t[is_suppr], y[is_suppr], label = "suppress", m=:x, ms=4, color=:red)
            end
        end
    end

    return plt
end

# small convenience: show a combined plot of all tasks for a bee on a single axes
function plot_tasks_combined_for_bee(df::DataFrame, bee; beecol::Symbol=:bee1_id, timecol::Symbol=:time, task_prefix::String="task_", smooth::Bool=false, smooth_k::Int=5)
    cols = task_columns(df; prefix=task_prefix)
    rows = df[df[!, beecol] .== bee, :]
    sort!(rows, timecol)
    plt = plot(title = "All tasks — bee $(bee)", xlabel = String(timecol), ylabel = "task value", legend = :right)
    for c in cols
        y = Float64.(rows[!, c])
        yplot = smooth ? _movavg(y, smooth_k) : y
        plot!(plt, rows[!, timecol], yplot, label = String(c))
    end
    return plt
end

export task_columns, plot_tasks_grid_for_bee, plot_task_grid_across_bees, plot_tasks_combined_for_bee
using DataFrames, Plots

# Utility: detect task columns (default prefix "task_")
function task_columns(df::DataFrame; prefix::String = "task_")
    pat = Regex("^" * prefix * raw"\d+$")
    # use `match` for compatibility: returns `nothing` if no match
    filter(c -> !isnothing(match(pat, String(c))), names(df))
end

# Plot the time evolution of all task columns for a single bee id.
# - df: your DataFrame
# - bee: the bee id to plot (matches column 'bee1_id' by default)
# - beecol: symbol name of bee id column (default :bee1_id)
# - timecol: symbol name of time column (default :time)
# - savepath: optional path to save PNG
function plot_tasks_for_bee(df::DataFrame, bee;
                            beecol::Symbol = :bee1_id,
                            timecol::Symbol = :time,
                            prefix::String = "task_",
                            savepath::Union{Nothing,String}=nothing)

    cols = task_columns(df; prefix=prefix)
    rows = df[df[!, beecol] .== bee, :]
    if nrow(rows) == 0
        error("No rows found for $beecol == $bee")
    end
    sort!(rows, timecol)

    plt = plot(title = "Tasks over time — bee $(bee)",
               xlabel = String(timecol), ylabel = "task value",
               legend = :outerright)
    for c in cols
        plot!(plt, rows[!, timecol], rows[!, c], label = String(c))
    end

    if savepath !== nothing
        savefig(plt, savepath)
    end
    return plt
end

# Plot the time evolution of one task (task index or column name) across all bees.
# - task: either an Int (task index) or Symbol (column name)
# - beecol: which column identifies the bee (default :bee1_id)
function plot_task_across_bees(df::DataFrame, task;
                               beecol::Symbol = :bee1_id,
                               timecol::Symbol = :time,
                               prefix::String = "task_",
                               savepath::Union{Nothing,String}=nothing)

    # resolve task column
    taskcol = task isa Int ? Symbol("$(prefix)$(task)") :
              task isa AbstractString ? Symbol(task) :
              task

    if !(taskcol in names(df))
        error("Task column $taskcol not found in DataFrame")
    end

    bees = sort(unique(df[!, beecol]))
    plt = plot(title = "Task $(taskcol) over time for each bee",
               xlabel = String(timecol), ylabel = String(taskcol),
               legend = :right)

    for b in bees
        rows = df[df[!, beecol] .== b, :]
        if nrow(rows) == 0
            continue
        end
        sort!(rows, timecol)
        plot!(plt, rows[!, timecol], rows[!, taskcol], label = "bee $b")
    end

    if savepath !== nothing
        savefig(plt, savepath)
    end
    return plt
end

# Optional: simple moving average smoother (for noisy time-series)
function movavg(x::AbstractVector, k::Int=5)
    if k <= 1
        return x
    end
    n = length(x)
    pad = floor(Int, k÷2)
    y = similar(x, Float64)
    for i in 1:n
        lo = max(1, i - pad); hi = min(n, i + pad)
        y[i] = mean(skipmissing(x[lo:hi]))
    end
    return y
end


using Makie

# Makie-based plot: all tasks for one bee (lines) with per-point markers indicating
# event type:
#  - :circle for unaffected points
#  - :rect (square) for training events (bee1==bee2 on that task)
#  - :utriangle for suppression events (bee1!=bee2 on that task)
#
# Usage:
# fig = plot_tasks_with_event_markers_makie(df, bee; kwargs...)
function plot_tasks_with_event_markers_makie(df::DataFrame, bee;
                                             beecol::Symbol = :bee1_id,
                                             otherbeecol::Symbol = :bee2_id,
                                             timecol::Symbol = :time,
                                             task_prefix::String = "task_",
                                             taskidcol::Symbol = :task_id,
                                             figsize = (1200, 400),
                                             linewidth = 2,
                                             markersize = 10)

    cols = task_columns(df; prefix=task_prefix)
    if isempty(cols)
        error("No task columns found with prefix='$(task_prefix)'")
    end

    rows = df[df[!, beecol] .== bee, :]
    if nrow(rows) == 0
        error("No rows found for $beecol == $bee")
    end
    sort!(rows, timecol)

    times = rows[!, timecol]

    # simple color palette
    n = length(cols)
    palette = [RGBf0(cos(i*0.6)%1, sin(i*0.8)%1, 0.5 + 0.5*sin(i*0.3)) for (i,_) in enumerate(cols)]

    fig = Makie.Figure(resolution = (figsize[1], figsize[2]))
    ax = Makie.Axis(fig[1, 1]; xlabel = String(timecol), ylabel = "task value",
                   title = "Bee $(bee) — tasks over time")

    # draw each task line and scatter markers with appropriate shapes
    for (i, col) in enumerate(cols)
        y = Float64.(rows[!, col])
        Makie.lines!(ax, times, y; color = palette[i], linewidth = linewidth)

        # determine per-point event types for this task
        taskidx = begin
            m = match(r"$(task_prefix)(\d+)", String(col))
            m === nothing ? nothing : parse(Int, m.captures[1])
        end

        if taskidx === nothing || !(taskidcol in names(rows))
            # no event markers available: mark all as circles
            Makie.scatter!(ax, times, y; color = palette[i], markersize = markersize, marker = :circle)
            continue
        end

        is_task = rows[!, taskidcol] .== taskidx
        is_train = is_task .& (rows[!, beecol] .== rows[!, otherbeecol])
        is_suppr = is_task .& (rows[!, beecol] .!= rows[!, otherbeecol])
        is_none = .!is_task

        if any(is_none)
            Makie.scatter!(ax, times[is_none], y[is_none]; color = palette[i], markersize = markersize, marker = :circle)
        end
        if any(is_train)
            Makie.scatter!(ax, times[is_train], y[is_train]; color = :green, markersize = markersize+2, marker = :rect)
        end
        if any(is_suppr)
            Makie.scatter!(ax, times[is_suppr], y[is_suppr]; color = :red, markersize = markersize+2, marker = :utriangle)
        end
    end

    # legend: create small legend markers
    labels = [String(c) for c in cols]
    # Add custom legend by plotting invisible points with labels
    for (i, lab) in enumerate(labels)
        Makie.lines!(ax, [NaN], [NaN]; color = palette[i], linewidth = linewidth, label = lab)
    end
    Makie.AxisLegend(fig[1,2], ax; position = :rt)

    return fig
end

export plot_tasks_with_event_markers_makie

# Draw tasks for a single bee into an existing Makie Axis (reusable helper)
function draw_tasks_on_axis!(ax::Makie.Axis, rows::DataFrame, cols::Vector{Symbol};
                            beecol::Symbol = :bee1_id,
                            otherbeecol::Symbol = :bee2_id,
                            timecol::Symbol = :time,
                            taskidcol::Symbol = :task_id,
                            palette = nothing,
                            linewidth::Real = 2,
                            markersize::Real = 10)

    if nrow(rows) == 0
        return ax
    end
    sort!(rows, timecol)
    times = rows[!, timecol]
    n = length(cols)
    if palette === nothing
        base_colors = (:blue, :orange, :green, :purple, :brown, :pink, :gray, :red)
        palette = [ base_colors[(i-1) % length(base_colors) + 1] for i in 1:n ]
    end

    for (i, col) in enumerate(cols)
        y = Float64.(rows[!, col])
        Makie.lines!(ax, times, y; color = palette[i], linewidth = linewidth)

    m = match(r"^task_(\d+)$", String(col))
        taskidx = m === nothing ? nothing : parse(Int, m.captures[1])

        if taskidx === nothing || !(taskidcol in names(rows))
            Makie.scatter!(ax, times, y; color = palette[i], markersize = markersize, marker = :circle)
            continue
        end

        is_task = rows[!, taskidcol] .== taskidx
        is_train = is_task .& (rows[!, beecol] .== rows[!, otherbeecol])
        is_suppr = is_task .& (rows[!, beecol] .!= rows[!, otherbeecol])
        is_none = .!is_task

        if any(is_none)
            Makie.scatter!(ax, times[is_none], y[is_none]; color = palette[i], markersize = markersize, marker = :circle)
        end
        if any(is_train)
            Makie.scatter!(ax, times[is_train], y[is_train]; color = :green, markersize = markersize+2, marker = :rect)
        end
        if any(is_suppr)
            Makie.scatter!(ax, times[is_suppr], y[is_suppr]; color = :red, markersize = markersize+2, marker = :utriangle)
        end
    end
    return ax
end


# Create a stacked Makie Figure with one axis per bee and draw each bee's tasks into it.
function plot_bees_stacked_makie(df::DataFrame, bees::Vector{<:Integer};
                                 beecol::Symbol = :bee1_id,
                                 task_prefix::String = "task_",
                                 timecol::Symbol = :time,
                                 taskidcol::Symbol = :task_id,
                                 figsize = (1200, 300))

    cols = task_columns(df; prefix=task_prefix)
    if isempty(cols)
        error("No task columns found with prefix='$(task_prefix)'")
    end

    n = length(bees)
    fig = Makie.Figure(resolution = (figsize[1], figsize[2]*n))
    axes = [Makie.Axis(fig[i, 1]; xlabel = i==n ? String(timecol) : "", ylabel = i==1 ? "task value" : "") for i in 1:n]

    # compute global time limits so every axis shares the same x-range
    if timecol in names(df) && nrow(df) > 0
        tmin = minimum(skipmissing(df[!, timecol])); tmax = maximum(skipmissing(df[!, timecol]))
    else
        tmin, tmax = nothing, nothing
    end

    for (i, b) in enumerate(bees)
        rows = df[df[!, beecol] .== b, :]
        draw_tasks_on_axis!(axes[i], rows, cols; beecol=beecol, otherbeecol=:bee2_id, timecol=timecol, taskidcol=taskidcol)
        if tmin !== nothing
            axes[i].xlimits = (tmin, tmax)
        end
        Makie.title!(axes[i], "Bee $(b)")
    end

    return fig
end

export draw_tasks_on_axis!, plot_bees_stacked_makie

 # Makie version — put this in a code cell
using DataFrames
using Makie, CairoMakie   # choose appropriate backend; load one (CairoMakie used here)

function task_columns(df::DataFrame; prefix::String="task_")
    pat = Regex("^" * prefix * raw"(\d+)$")
    filter(c -> ismatch(pat, String(c)), names(df))
end

function plot_tasks_for_bee_makie(df::DataFrame, bee;
                                  beecol::Symbol=:bee1_id,
                                  otherbeecol::Symbol=:bee2_id,
                                  timecol::Symbol=:time,
                                  task_prefix::String="task_",
                                  taskidcol::Symbol=:task_id,
                                  figsize=(1200,400),
                                  linewidth=2,
                                  markersize=10)

    # find task columns and per-column integer index
    cols = task_columns(df; prefix=task_prefix)
    if isempty(cols)
        error("No task columns like $(task_prefix)N found")
    end

    # subset rows for this bee and sort by time
    rows = df[df[!, beecol] .== bee, :]
    if nrow(rows) == 0
        error("No rows for $beecol == $bee")
    end
    sort!(rows, timecol)
    times = rows[!, timecol]

    # palette (cycle through a small set of colors)
    base_colors = [:blue, :orange, :green, :purple, :brown, :pink, :gray, :red]
    palette = [ base_colors[(i-1) % length(base_colors) + 1] for i in 1:length(cols) ]

    # precompute task index for each column (or nothing)
    pat = Regex("^" * task_prefix * raw"(\d+)$")
    taskidx_map = Dict{Symbol,Union{Int,Nothing}}()
    for c in cols
        m = match(pat, String(c))
        taskidx_map[c] = m === nothing ? nothing : parse(Int, m.captures[1])
    end

    fig = Figure(resolution = (figsize[1], figsize[2]))
    ax = Axis(fig[1, 1]; xlabel = String(timecol), ylabel = "task value",
              title = "Bee $(bee) — tasks over time")

    # draw lines + per-point markers
    for (i, col) in enumerate(cols)
        y = Float64.(rows[!, col])
        lines!(ax, times, y; color = palette[i], linewidth = linewidth)

        taskidx = taskidx_map[col]
        if taskidx === nothing || !(taskidcol in names(rows))
            # no per-row task info: mark all points as circles
            scatter!(ax, times, y; color = palette[i], markersize = markersize, marker = :circle)
            continue
        end

        # classify events (per-row booleans)
        is_task = rows[!, taskidcol] .== taskidx
        is_train = is_task .& (rows[!, beecol] .== rows[!, otherbeecol])
        is_suppr = is_task .& (rows[!, beecol] .!= rows[!, otherbeecol])
        is_none = .!is_task

        if any(is_none)
            scatter!(ax, times[is_none], y[is_none]; color = palette[i], markersize = markersize, marker = :circle)
        end
        if any(is_train)
            scatter!(ax, times[is_train], y[is_train]; color = :green, markersize = markersize+2, marker = :rect)
        end
        if any(is_suppr)
            scatter!(ax, times[is_suppr], y[is_suppr]; color = :red, markersize = markersize+2, marker = :utriangle)
        end
    end

    # small legend: create proxy lines for the legend
    for (i, col) in enumerate(cols)
        label = String(col)
        lines!(ax, [NaN], [NaN]; color = palette[i], linewidth = linewidth, label = label)
    end
    legend = Legend(fig, ax; orientation = :vertical)
    fig[1, 2] = legend

    return fig
end

# Example usage (CairoMakie backend):
# df = CSV.read("path/to/log.csv", DataFrame)
# fig = plot_tasks_for_bee_makie(df, 1)
# save("bee1_tasks_makie.png", fig)   # or display(fig) in Jupyter/Quarto






function plot_bee_all_tasks(df::DataFrame, bee_id;
                                  figsize=(1200,400),
                                  linewidth=2,
                                  markersize=10,
                                  mapping::Union{Nothing,Dict{Any,String}}=nothing, 
                                  add_markers::Bool=true)
    tasks = sort(unique(df.task_id))
    
    fig = Makie.Figure(resolution = (figsize[1], figsize[2]))
    ax = Makie.Axis(fig[1, 1], 
              xlabel = "Time", 
              ylabel = "Accuracy",
              title = "Evolution of Accuracy for Bee $bee_id Across All Tasks")
    
    colors = Makie.wong_colors()[1:length(tasks)]
    
    # Plot individual points with markers
    training_df = filter(row -> row.bee1_id == bee_id && row.bee1_id == row.bee2_id, df)
    suppression_df = filter(row -> row.bee1_id == bee_id && row.bee1_id != row.bee2_id, df)

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
            Makie.scatter!(ax, train_rows.time, ytrain, 
                    color = :green, markersize = markersize+2, marker = :rect)
        end
        # Plot suppression events
        suppr_rows = suppression_df[suppression_df.task_id .== task, :]
        if nrow(suppr_rows) > 0 && add_markers
            ysuppr = Float64.(suppr_rows[!, col_name])
            Makie.scatter!(ax, suppr_rows.time, ysuppr, 
                    color = :red, markersize = markersize+2, marker = :utriangle)
        end
    end

    # Add legend
    axislegend(ax, position = :lt)
    
    # Add grid
    ax.xgridvisible = true
    ax.ygridvisible = true
    return fig
end


function plot_bees_stacked_makie(df::DataFrame, bees::Vector{<:Integer};
                                 beecol::Symbol = :bee1_id,
                                 task_prefix::String = "task_",
                                 timecol::Symbol = :time,
                                 taskidcol::Symbol = :task_id,
                                 figsize = (1200, 300))

    cols = task_columns(df; prefix=task_prefix)
    if isempty(cols)
        error("No task columns found with prefix='$(task_prefix)'")
    end

    n = length(bees)
    fig = Makie.Figure(size = (figsize[1], figsize[2]*n))
    axes = [Makie.Axis(fig[i, 1]; xlabel = i==n ? String(timecol) : "", ylabel = i==1 ? "task value" : "") for i in 1:n]

    # compute global time limits so every axis shares the same x-range
    if timecol in names(df) && nrow(df) > 0
        tmin = minimum(skipmissing(df[!, timecol])); tmax = maximum(skipmissing(df[!, timecol]))
    else
        tmin, tmax = nothing, nothing
    end

    for (i, b) in enumerate(bees)
    # draw directly from the full df for this bee (draw_tasks_on_axis! will subset)
    draw_tasks_on_axis!(axes[i], df, b; beecol=beecol, otherbeecol=:bee2_id, timecol=timecol, task_prefix=task_prefix, taskidcol=taskidcol)
        if tmin !== nothing
            axes[i].xlimits = (tmin, tmax)
        end
        Makie.title!(axes[i], "Bee $(b)")
    end

    return fig
end

# Draw tasks for a single bee into an existing Makie Axis (reusable helper)
# Signature mirrors plot_tasks_for_bee_makie to make it easy to read and adapt
function draw_tasks_on_axis!(ax::Makie.Axis, df::DataFrame, bee;
                            beecol::Symbol = :bee1_id,
                            otherbeecol::Symbol = :bee2_id,
                            timecol::Symbol = :time,
                            task_prefix::String = "task_",
                            taskidcol::Symbol = :task_id,
                            palette = nothing,
                            linewidth::Real = 2,
                            markersize::Real = 10)

    # select rows for this bee and find task columns
    rows = df[df[!, beecol] .== bee, :]
    if nrow(rows) == 0
        return ax
    end
    cols = task_columns(df; prefix=task_prefix)
    sort!(rows, timecol)
    times = rows[!, timecol]

    n = length(cols)
    if palette === nothing
        base_colors = (:blue, :orange, :green, :purple, :brown, :pink, :gray, :red)
        palette = [ base_colors[(i-1) % length(base_colors) + 1] for i in 1:n ]
    end

    for (i, col) in enumerate(cols)
        y = Float64.(rows[!, col])
        Makie.lines!(ax, times, y; color = palette[i], linewidth = linewidth)

        # extract numeric task index from column name like "task_3"
    pat = Regex("^" * task_prefix * raw"(\\d+)$")
    m = match(pat, String(col))
    taskidx = m === nothing ? nothing : parse(Int, m.captures[1])

        if taskidx === nothing || !(taskidcol in names(rows))
            Makie.scatter!(ax, times, y; color = palette[i], markersize = markersize, marker = :circle)
            continue
        end

        is_task = rows[!, taskidcol] .== taskidx
        is_train = is_task .& (rows[!, beecol] .== rows[!, otherbeecol])
        is_suppr = is_task .& (rows[!, beecol] .!= rows[!, otherbeecol])
        is_none = .!is_task

        if any(is_none)
            Makie.scatter!(ax, times[is_none], y[is_none]; color = palette[i], markersize = markersize, marker = :circle)
        end
        if any(is_train)
            Makie.scatter!(ax, times[is_train], y[is_train]; color = :green, markersize = markersize+2, marker = :rect)
        end
        if any(is_suppr)
            Makie.scatter!(ax, times[is_suppr], y[is_suppr]; color = :red, markersize = markersize+2, marker = :utriangle)
        end
    end
    return ax
end


using CairoMakie, ColorSchemes, DataFrames, Statistics

"""
plot_specialization_heatmap(log::DataFrame; kwargs...)

Create a heatmap of final per-bee × per-task values and annotate each bee's best task.

Inputs
- log::DataFrame : event log with time, bee1_id, bee2_id, task_id and columns `task_N` (post-event values)
Keyword args
- task_prefix::String = "task_"         : prefix of task columns
- timecol::Symbol = :time               : time column
- bee1col::Symbol = :bee1_id
- bee2col::Symbol = :bee2_id
- show_values::Bool = false             : overlay numeric values in cells
- dominance_threshold::Real = 0.0       : minimum normalized dominance to consider a 'strong' specialization
- cmap = :viridis                       : colormap symbol or ColorScheme
- nan_color = RGBAf0(0.8,0.8,0.8,1.0)   : color to use for NaNs (no-data)
- fig_size = (800, 400)

Returns a NamedTuple:
(figure, axis, bees, task_ids, matrix, per_bee_best_task, per_task_top_bee, per_bee_spec_score)
"""
function plot_specialization_heatmap(log::DataFrame; task_prefix::String="task_",
                                     timecol::Symbol=:time, bee1col::Symbol=:bee1_id,
                                     bee2col::Symbol=:bee2_id,
                                     show_values::Bool=false,
                                     dominance_threshold::Real=0.0,
                                     cmap=:viridis,
                                     nan_color = RGBAf0(0.8,0.8,0.8,1.0),
                                     fig_size=(800,400))

    # 1) discover task ids
    task_cols = filter(c -> startswith(string(c), task_prefix), names(log))
    task_ids = Int[]
    for c in task_cols
        m = match(Regex("^" * task_prefix * raw"(\d+)$"), string(c))
        if m !== nothing
            push!(task_ids, parse(Int, m.captures[1]))
        end
    end
    sort!(task_ids)
    n_tasks = length(task_ids)

    if n_tasks == 0
        error("plot_specialization_heatmap: no task columns found with prefix=$(task_prefix)")
    end

    # 2) collect bees and build last-known states (scan in time order)
    bees = unique(vcat(log[!, bee1col], log[!, bee2col]))
    bees = collect(bees)  # preserve order (you may want to sort)
    n_bees = length(bees)

    last_known = Dict{Any, Dict{Int,Float64}}()
    # initialize
    for b in bees
        last_known[b] = Dict{Int,Float64}()  # we'll fill tasks, missing -> NaN later
    end

    sorted_log = sort(log, timecol)
    for row in eachrow(sorted_log)
        b = row[bee1col]
        for tid in task_ids
            col = Symbol(task_prefix * string(tid))
            if (hasproperty(row, col) || haskey(row, col)) && !ismissing(row[col])
                last_known[b][tid] = float(row[col])
            end
        end
    end

    # fill missing with NaN so matrix is rectangular
    for b in bees
        for tid in task_ids
            if !haskey(last_known[b], tid)
                last_known[b][tid] = NaN
            end
        end
    end

    # 3) build matrix rows=bees, cols=tasks (Matrix{Float64})
    M = Array{Float64}(undef, n_bees, n_tasks)
    for (i,b) in enumerate(bees)
        for (j,tid) in enumerate(task_ids)
            M[i,j] = last_known[b][tid]
        end
    end

    # 4) compute per-bee best task and per-bee spec score (normalized advantage)
    eps = 1e-9
    per_bee_best_task = Dict{Any,Int}()
    per_bee_best_value = Dict{Any,Float64}()
    per_bee_spec_score = Dict{Any,Float64}()
    for (i,b) in enumerate(bees)
        vals = M[i, :]
        valid_idx = findall(!isnan, vals)
        if isempty(valid_idx)
            per_bee_best_task[b] = -1
            per_bee_best_value[b] = NaN
            per_bee_spec_score[b] = NaN
            continue
        end
        # find best and compute advantage over mean of others
        best_rel = argmax(vals[valid_idx])
        best_idx = valid_idx[best_rel]
        best_val = vals[best_idx]
        others = [vals[k] for k in 1:length(vals) if k != best_idx && !isnan(vals[k])]
        per_bee_best_task[b] = task_ids[best_idx]
        per_bee_best_value[b] = best_val
        if isempty(others) || isnan(best_val)
            per_bee_spec_score[b] = NaN
        else
            per_bee_spec_score[b] = (best_val - mean(others)) / (best_val + eps)
        end
    end

    # 5) per-task top bee (for overlay) and dominance
    per_task_top_bee = Dict{Int,Any}()
    per_task_top_value = Dict{Int,Float64}()
    per_task_second_value = Dict{Int,Float64}()
    for (j,tid) in enumerate(task_ids)
        col_vals = [(bees[i], M[i,j]) for i in 1:n_bees if !isnan(M[i,j])]
        if isempty(col_vals)
            per_task_top_bee[tid] = nothing
            per_task_top_value[tid] = NaN
            per_task_second_value[tid] = NaN
            continue
        end
        sorted_col = sort(col_vals, by=x->x[2], rev=true)
        per_task_top_bee[tid] = sorted_col[1][1]
        per_task_top_value[tid] = sorted_col[1][2]
        per_task_second_value[tid] = length(sorted_col) >= 2 ? sorted_col[2][2] : NaN
    end

    # 6) plotting
    fig = Figure(resolution = fig_size)  # (Makie accepts `resolution=` on recent versions; if backend warns, replace with size=)
    ax = Axis(fig[1,1]; xlabel="task id", ylabel="bee", xticks=(1:n_tasks, string.(task_ids)),
              yticks=(1:n_bees, string.(bees)), width=fig_size[1], height=fig_size[2])

    # handle NaNs: map NaNs to a color by creating a colormap with nan_color
    # simplest: replace NaN with minimum-δ so they appear distinct; keep a mask for annotations
    finite_vals = filter(!isnan, vec(M))
    if isempty(finite_vals)
        # nothing to show
        heatmap!(ax, fill(0.0, size(M)); colormap=cmap)
    else
        vmin = minimum(finite_vals)
        vmax = maximum(finite_vals)
        # create display matrix where NaN -> vmin - delta
        delta = max((vmax - vmin) * 0.05, 1e-6)
        M_disp = copy(M)
        nanmask = isnan.(M_disp)
        M_disp[nanmask] .= vmin - delta
        # choose a colormap and plot
        hm = heatmap!(ax, 1:n_tasks, 1:n_bees, M_disp';
                      colormap = cmap, colorrange=(vmin - delta, vmax))
        # set NaN color by adding a colorbar and drawing a small rectangle legend manually is more work;
        # we annotate NaNs below if show_values
    end

    # optionally overlay numeric values
    if show_values
        for i in 1:n_bees, j in 1:n_tasks
            val = M[i,j]
            if !isnan(val)
                text!(ax, j, i, string(round(val, sigdigits=3)); align = (:center, :center), color=:white, fontsize=10)
            else
                text!(ax, j, i, "—"; align = (:center, :center), color=:black, fontsize=8)
            end
        end
    end

    # overlay markers for per-task top bee and per-bee best task:
    for (j, tid) in enumerate(task_ids)
        b = per_task_top_bee[tid]
        if b === nothing
            continue
        end
        bi = findfirst(isequal(b), bees)
        if bi === nothing
            continue
        end
        # draw circle marker on the top bee for this task
        scatter!(ax, [j], [bi]; markersize=10, color=:white, strokewidth=2, strokecolor=:black)
    end

    # highlight bees' best tasks (different marker if dominance small)
    for (i,b) in enumerate(bees)
        tid = per_bee_best_task[b]
        if tid == -1
            continue
        end
        j = findfirst(==(tid), task_ids)
        if j === nothing
            continue
        end
        spec = per_bee_spec_score[b]
        color = isnan(spec) ? RGBAf0(0.5,0.5,0.5,1.0) : (spec >= dominance_threshold ? :red : :orange)
        scatter!(ax, [j], [i]; markersize = 14, color = color, marker = :star5, strokewidth=1.5, strokecolor=:black)
    end

    Colorbar(fig[1,2], ax.scene.plots[1]; label="task value")  # attach colorbar

    fig.layoutgap = (10,10)

    display(fig)

    return (figure = fig,
            axis = ax,
            bees = bees,
            task_ids = task_ids,
            matrix = M,
            per_bee_best_task = per_bee_best_task,
            per_bee_spec_score = per_bee_spec_score,
            per_task_top_bee = per_task_top_bee,
            per_task_top_value = per_task_top_value)
end