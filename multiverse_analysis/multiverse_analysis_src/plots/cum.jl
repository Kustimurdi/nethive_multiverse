using DataFrames

"""
For each task_id:
- create the union of all event times (train or suppress)
- compute cumulative counts for train and suppress on that union grid
Returns a DataFrame with columns: task_id, time, cum_train, cum_suppress
"""
function cum_curves_union_times(events::DataFrame)
    tasks = sort(unique(events.task_id))
    out = DataFrame(task_id=Int[], time=Float64[], cum_train=Int[], cum_suppress=Int[])

    for task in tasks
        e = events[events.task_id .== task, :]
        times = sort(unique(e.time))  # union grid for this task

        # counts per time per kind
        c = combine(groupby(e, [:time, :kind]), nrow => :n)
        # dictionaries for fast lookup (missing -> 0)
        train_counts = Dict{Float64, Int}()
        suppress_counts = Dict{Float64, Int}()
        for r in eachrow(c)
            if r.kind == "train"
                train_counts[r.time] = r.n
            else
                suppress_counts[r.time] = r.n
            end
        end

        ct = 0
        cs = 0
        for t in times
            ct += get(train_counts, t, 0)
            cs += get(suppress_counts, t, 0)
            push!(out, (task, t, ct, cs))
        end
    end

    return out
end

using CairoMakie

function plot_cum_events_per_task_union(curves::DataFrame; ncols=3)
    tasks = sort(unique(curves.task_id))
    nt = length(tasks)
    nrows = ceil(Int, nt / ncols)

    fig = Figure(resolution=(450*ncols + 220, 320*nrows + 120))

    # We create legend handles once (from the first subplot that has data)
    h_train = nothing
    h_supp  = nothing

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

        s = curves[curves.task_id .== task, :]
        isempty(s) && continue

        ht = lines!(ax, s.time, s.cum_train)
        hs = lines!(ax, s.time, s.cum_suppress, linestyle = :dash)

        if h_train === nothing
            h_train = ht
            h_supp  = hs
        end
    end

    Label(fig[nrows+1, 1:ncols], "time", tellwidth=false, fontsize=24)
    Label(fig[1:nrows, 0], "cumulative # events", rotation=pi/2, tellheight=false, fontsize=24)

    if h_train !== nothing
        Legend(fig[1:nrows, ncols+1], [h_train, h_supp], ["training", "suppression"])
    end

    colgap!(fig.layout, 14)
    rowgap!(fig.layout, 14)

    return fig
end

