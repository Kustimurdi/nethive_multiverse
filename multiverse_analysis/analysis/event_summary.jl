module EventSummary

using DataFrames, CSV, CairoMakie, Statistics
import CairoMakie: Figure, Axis, barplot, lines!, scatter!, save

"""Compute per-bee and per-task event counts (training/suppression).
Returns (per_bee_df, per_task_df).
"""
function compute_event_counts(log::DataFrame)
    per_bee = combine(groupby(log, :bee1_id)) do sub
        n_events = nrow(sub)
        n_training = sum(sub.bee1_id .== sub.bee2_id)
        n_suppression = n_events - n_training
        (n_events = n_events, n_training = n_training, n_suppression = n_suppression,
         frac_training = n_training / max(n_events, 1), frac_suppression = n_suppression / max(n_events,1))
    end

    per_task = combine(groupby(log, :task_id)) do sub
        n_events = nrow(sub)
        n_training = sum(sub.bee1_id .== sub.bee2_id)
        n_suppression = n_events - n_training
        (n_events = n_events, n_training = n_training, n_suppression = n_suppression,
         frac_training = n_training / max(n_events,1), frac_suppression = n_suppression / max(n_events,1))
    end

    return per_bee, per_task
end

"""Save CSV summaries and simple bar plots into `outdir`. Creates `outdir` if needed.
Returns a Dict with file paths created and the DataFrames.
"""
function save_event_summaries(log::DataFrame; outdir::AbstractString="multiverse_analysis/analysis/output")
    # make directory
    isdir(outdir) || mkpath(outdir)

    per_bee, per_task = compute_event_counts(log)

    # CSV outputs
    per_bee_csv = joinpath(outdir, "per_bee_counts.csv")
    per_task_csv = joinpath(outdir, "per_task_counts.csv")
    CSV.write(per_bee_csv, per_bee)
    CSV.write(per_task_csv, per_task)

    # Simple bar plots (sorted descending by n_events)
    sort!(per_bee, :n_events, rev=true)
    sort!(per_task, :n_events, rev=true)

    # per-bee bar
    fig1 = Figure(resolution=(900,400))
    ax1 = Axis(fig1[1,1]; xlabel = "Bee (bee1_id)", ylabel = "Events", title = "Events per Bee (bee1)")
    bees = string.(per_bee.bee1_id)
    barplot!(ax1, 1:length(bees), per_bee.n_events; color = :steelblue)
    xticks!(ax1, 1:length(bees), bees; rotation=45)
    per_bee_png = joinpath(outdir, "per_bee_events.png")
    save(per_bee_png, fig1)

    # per-task bar
    fig2 = Figure(resolution=(900,400))
    ax2 = Axis(fig2[1,1]; xlabel = "Task ID", ylabel = "Events", title = "Events per Task")
    tasks = string.(per_task.task_id)
    barplot!(ax2, 1:length(tasks), per_task.n_events; color = :seagreen)
    xticks!(ax2, 1:length(tasks), tasks; rotation=0)
    per_task_png = joinpath(outdir, "per_task_events.png")
    save(per_task_png, fig2)

    return Dict(
        :per_bee_csv => per_bee_csv,
        :per_task_csv => per_task_csv,
        :per_bee_png => per_bee_png,
        :per_task_png => per_task_png,
        :per_bee_df => per_bee,
        :per_task_df => per_task
    )
end

end # module

export EventSummary
