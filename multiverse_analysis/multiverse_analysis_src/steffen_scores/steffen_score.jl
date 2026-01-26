include("/scratch/n/N.Pfaffenzeller/nikolas_nethive/nethive_multiverse/multiverse_analysis/multiverse_analysis_src/onset/onset_analysis.jl")

function state_at_time(acc_log::DataFrame, t::Real;
                       agent_col::Symbol = :bee1_id,
                       time_col::Symbol  = :time,
                       task_regex = r"^task_\d+$")
    # task columns
    task_cols = Symbol.(filter(c -> occursin(task_regex, String(c)), names(acc_log)))

    # keep only rows up to time t
    sub = acc_log[acc_log[!, time_col] .<= t, :]

    # if an agent never appeared before time t, it will be missing from the state
    isempty(sub) && return DataFrame()

    # last-known row per agent (LOCF) within sub
    last_rows = unique(
        sort(sub, [agent_col, time_col], rev = [false, true]),
        agent_col
    )

    # keep only what you care about
    select!(last_rows, [agent_col, time_col, task_cols...])
    sort!(last_rows, agent_col)

    return last_rows
end

task_cols(state) = filter(c -> occursin(r"^task_\d+$", String(c)), names(state))

function task_matrix(state::DataFrame)
    tcols = task_cols(state)
    A = Matrix{Float64}(state[:, tcols])   # size: n_agents × n_tasks
    println(A)
    return A, tcols
end

function n_tasks_with_any_geq(state::DataFrame, q::Real)
    A, _ = task_matrix(state)
    # for each task (column), check if any agent meets threshold
    count_task = count(j -> any(x -> isfinite(x) && x ≥ q, view(A, :, j)), 1:size(A,2))
    return count_task#, count_task/size(A,2)
end

function avg_agents_geq_per_task(state::DataFrame, q::Real)
    A, _ = task_matrix(state)

    counts_per_task = [
        count(x -> isfinite(x) && x >= q, view(A, :, j))
        for j in 1:size(A, 2)
    ]

    return mean(counts_per_task)
end

function n_tasks_with_exactly_one_geq(state::DataFrame, q::Real)
    A, _ = task_matrix(state)
    count_task = count(j -> begin
        n = count(x -> isfinite(x) && x ≥ q, view(A, :, j))
        n == 1
    end, 1:size(A,2))
    return count_task#, count_task/size(A,2)
end


function n_tasks_with_unique_specialist(state::DataFrame, q::Real)
    A, _ = task_matrix(state)
    n_agents, n_tasks = size(A)

    count_task = 0
    for j in 1:n_tasks
        # which agents meet threshold on task j?
        good = [i for i in 1:n_agents if isfinite(A[i,j]) && A[i,j] ≥ q]
        length(good) == 1 || continue
        i = only(good)

        v = A[i,j]
        # specialist condition: all other task scores strictly lower than v
        is_specialist = all(k -> (k == j) || (!isfinite(A[i,k]) ? true : A[i,k] < v), 1:n_tasks)

        if is_specialist
            count_task += 1
        end
    end
    return count_task#, count_task/n_tasks
end

function task_coverage_stats(state::DataFrame, q::Real)
    return (
        any_geq = n_tasks_with_any_geq(state, q),
        exactly_one_geq = n_tasks_with_exactly_one_geq(state, q),
        unique_specialist = n_tasks_with_unique_specialist(state, q),
        avg_agents_geq_per_task = avg_agents_geq_per_task(state, q)
    )
end

function steffen_scores_over_time(acc_log::DataFrame; time_points::AbstractVector{<:Real}, agent_col::Symbol = :bee1_id,
                                  time_col::Symbol  = :time,
                                  task_regex = r"^task_\d+$",
                                  thresholds = [0.8, 0.9])
    results = Vector{NamedTuple}(undef, length(time_points))

    for (idx, t) in enumerate(time_points)
        state = state_at_time(acc_log, t;
                              agent_col=agent_col,
                              time_col=time_col,
                              task_regex=task_regex)

        stats_per_threshold = Dict{Float64, NamedTuple}()
        for q in thresholds
            stats_per_threshold[q] = task_coverage_stats(state, q)
        end

        results[idx] = (
            time = t,
            n_agents = nrow(state),
            n_tasks = length(task_cols(state)),
            stats_per_threshold = stats_per_threshold,
        )
    end

    return results
end

function steffen_scores_over_time_df(acc_log::DataFrame;
    time_points::AbstractVector{<:Real},
    agent_col::Symbol = :bee1_id,
    time_col::Symbol  = :time,
    task_regex = r"^task_\d+$",
    thresholds = [0.8, 0.9]
)
    out = DataFrame(
        time = Float64[],
        threshold = Float64[],
        n_agents = Int[],
        n_tasks = Int[],
        any_geq = Int[],
        exactly_one_geq = Int[],
        unique_specialist = Int[],
    )

    for t in time_points
        state = state_at_time(acc_log, t;
            agent_col = agent_col,
            time_col  = time_col,
            task_regex = task_regex
        )

        na = nrow(state)
        nt = length(task_cols(state))

        for q in thresholds
            stats = task_coverage_stats(state, q)
            push!(out, (
                float64(t),
                float64(q),
                na,
                nt,
                stats.any_geq,
                stats.exactly_one_geq,
                stats.unique_specialist
            ))
        end
    end

    return out
end

function steffen_scores_one_run(run; thresholds=[0.8, 0.9],
                                time_points=nothing)
    # sanity checks
    isfile(run.config_path)    || error("missing config.json")
    isfile(run.data_path) || error("missing data log, prob. event_log.csv")

    # --- load only selected config values (as a 1-row DF) ---
    cfg_all = config_to_df(run.config_path)  # 1-row DF
    keep_cfg = ["n_epochs", "learning_rate", "interaction_rate", "dead_time", "lambda_sensitivity"]

    # keep only columns that actually exist (avoids crashes if one key missing)
    cfg = select(cfg_all, intersect(keep_cfg, names(cfg_all)))

    # --- load event log (DataFrame) ---
    acc_log = CSV.read(run.event_log_path, DataFrame)  # adjust if you already have a loader

    if time_points === nothing
        tmax = parse(Float64, cfg_all[1, :n_epochs])  # assuming n_epochs corresponds to max time
        time_points = [tmax / 2, tmax]
    end

    # --- compute analysis (many rows) ---
    ana = steffen_scores_over_time_df(acc_log;
        time_points=time_points,
        thresholds=thresholds
    )
    n = nrow(ana)

    # --- repeat cfg + metadata to match ana rows ---
    # repeat the 1-row cfg to n rows
    cfg_rep = repeat(cfg, n)

    # metadata columns (length n)
    meta = DataFrame(
        root          = fill(run.root, n),
        param_folder  = fill(run.param_folder, n),
        param_dir     = fill(run.param_dir, n),
        #event_log_path = fill(run.event_log_path, n),
        #config_path    = fill(run.config_path, n),
    )

    # final merged table
    out = hcat(meta, cfg_rep, ana; makeunique=true)

    return out
end

function steffen_scores_all_runs_from_config(roots::Vector{String};
    thresholds = [0.8, 0.9],
    time_points = nothing,                 # NEW
    event_filename = "event_log.csv",
    #event_filename = "states.csv",        # UPDATED
    config_filename = "config.json",
)
    runs = collect_runs(roots; event_filename=event_filename, config_filename=config_filename)
    n = length(runs)

    results_vec = Vector{Union{Nothing,DataFrame}}(fill(nothing, n))
    failures = Vector{NamedTuple}(undef, 0)
    fail_lock = ReentrantLock()

    p = Progress(n; desc = "Analyzing runs")

    Threads.@threads for i in 1:n
        run = runs[i]
        try
            results_vec[i] = steffen_scores_one_run(run;
                thresholds = thresholds,
                time_points = time_points
            )
        catch e
            lock(fail_lock) do
                push!(failures, (
                    root = run.root,
                    param_folder = run.param_folder,
                    param_dir = run.param_dir,
                    #event_log_path = run.event_log_path,
                    #config_path = run.config_path,
                    error = sprint(showerror, e),
                    # optional: include backtrace for debugging
                    # backtrace = sprint(Base.show_backtrace, catch_backtrace()),
                ))
            end
        end

        next!(p)

    end

    rows = [df for df in results_vec if df !== nothing]
    results = isempty(rows) ? DataFrame() : vcat(rows...; cols=:union)
    failures_df = isempty(failures) ? DataFrame() : DataFrame(failures)

    return results, failures_df
end