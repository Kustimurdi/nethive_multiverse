using DataFrames, Statistics

function mean_dt_per_agent_task(event_log::DataFrame)
    # Minimal event table + classify kind
    filter!(row -> row.task_id != 0, event_log)  # remove task_id == 0
    e = select(event_log, :time, :bee1_id, :bee2_id, :task_id)
    e.kind  = ifelse.(e.bee1_id .== e.bee2_id, "train", "suppress")
    e.agent = e.bee1_id

    sort!(e, [:agent, :task_id, :kind, :time])

    # compute Δt within each (agent, task, kind)
    g = groupby(e, [:agent, :task_id, :kind])

    dt_df = combine(g) do sdf
        ts = sdf.time
        n  = length(ts)
        if n < 2
            # not enough events to define an inter-event time
            return DataFrame(
                n_events = n,
                mean_dt  = missing,
                std_dt   = missing,
                var_dt   = missing
            )
        end
        dts = diff(ts)
        return DataFrame(
            n_events = n,
            mean_dt  = mean(dts),
            std_dt   = std(dts),        # sample std
            var_dt   = var(dts)         # sample variance
        )
    end

    return dt_df
end

function plot_dt_per_agent(dt_df::DataFrame; ncols=3, sharey=true)
    cols = Makie.wong_colors()         # nice default palette
    train_col = cols[1]
    supp_col  = cols[2]

    legend_elems = [
        PolyElement(color=train_col),
        PolyElement(color=supp_col)
    ]
    legend_labels = ["training", "suppression"]

    agents = sort(unique(dt_df.agent))
    nA = length(agents)
    nrows = ceil(Int, nA / ncols)

    tasks = sort(unique(dt_df.task_id))  # global task order
    kinds = ["train", "suppress"]

    ylims_all = nothing
    if sharey
        vals = skipmissing(dt_df.mean_dt)
        if !isempty(vals)
            ymax = maximum(collect(vals .+ coalesce.(dt_df.std_dt, 0.0)))
            ylims_all = (0.0, ymax * 1.05)
        end
    end

    fig = Figure(resolution=(520*ncols + 220, 380*nrows + 120))

    h_train = nothing
    h_supp  = nothing

    for (k, agent) in enumerate(agents)
        r = div(k-1, ncols) + 1
        c = mod(k-1, ncols) + 1

        ax = Axis(fig[r, c],
            title = "agent $(agent)",
            xlabelvisible = false,
            ylabelvisible = false,
            xlabel = "task",
            ylabel = "mean Δt",
            xticklabelsize = 22,
            yticklabelsize = 22,
            titlesize = 18,
            xlabelsize = 20,
            ylabelsize = 20,
        )
        if ylims_all !== nothing
            ylims!(ax, ylims_all...)
        end

        sub = dt_df[dt_df.agent .== agent, :]

        x = 1:length(tasks)
        offset = 0.06
        #width = 0.32

        for kind in kinds
            s = sub[sub.kind .== kind, :]
            isempty(s) && continue

            mean_map = Dict(s.task_id .=> s.mean_dt)
            std_map  = Dict(s.task_id .=> coalesce.(s.std_dt, 0.0))

            y = [haskey(mean_map, t) ? Float64(mean_map[t]) : NaN for t in tasks]
            e = [haskey(std_map,  t) ? Float64(std_map[t])  : NaN for t in tasks]

            xj = (kind == "train") ? (x .- offset) : (x .+ offset)
            idx = findall(isfinite, y)

            col = (kind == "train") ? train_col : supp_col
            #hb = barplot!(ax, xj[idx], y[idx]; width=width ,color = col)
            scatter!(ax, xj[idx], y[idx];
                color = col,
                markersize = 14)

            errorbars!(ax, xj[idx], y[idx], e[idx]; color = col, whiskerwidth = 10)

            #if h_train === nothing && kind == "train"
                #h_train = hb
            #elseif h_supp === nothing && kind == "suppress"
                #h_supp = hb
            #end
        end

        ax.xticks = (1:length(tasks), string.(tasks))
    end

    Label(fig[nrows+1, 1:ncols],
        "agent id",          # or "task id" for the other layout
        tellwidth = false,
        fontsize = 28)

    Label(fig[1:nrows, 0],
        "mean Δt",
        rotation = pi/2,
        tellheight = false,
        fontsize = 28)


    if h_train !== nothing && h_supp !== nothing
        #Legend(fig[1:nrows, ncols+1], [h_train, h_supp], ["training", "suppression"])
        Legend(fig[1:nrows, ncols+1], legend_elems, legend_labels)
    end

    colgap!(fig.layout, 14)
    rowgap!(fig.layout, 14)

    return fig
end
